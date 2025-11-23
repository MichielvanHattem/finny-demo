import streamlit as st
import pandas as pd
import os
import json
import re
import glob
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# 1. CONFIGURATIE & STATE
st.set_page_config(page_title="Finny", layout="wide")
load_dotenv()

# Initialiseer Geheugen
if "active_years" not in st.session_state:
    st.session_state["active_years"] = [2024] # Default
if "messages" not in st.session_state: 
    st.session_state.messages = []

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="pw", on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# 2. DATA LADEN
@st.cache_data
def load_data():
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": ""}
    
    def clean_code(val):
        return str(val).split('.')[0].strip()

    # A. TRANSACTIES
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", dtype=str)
            
            # Bedragen naar float
            if 'AmountDC_num' in df.columns:
                df['AmountDC_num'] = df['AmountDC_num'].str.replace(',', '.').replace('nan', '0')
                df['AmountDC_num'] = pd.to_numeric(df['AmountDC_num'], errors='coerce').fillna(0.0)
            
            # Codes & Jaren cleanen
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            if 'Finny_Year' in df.columns:
                # Zorg dat jaren integers zijn voor filteren
                df['Year_Clean'] = df['Finny_Year'].astype(str).str.split('.').str[0]

            # Universal Search kolom (voor Lookups)
            search_cols = ['Description', 'AccountName', 'Finny_GLDescription', 'Finny_GLCode']
            existing_cols = [c for c in search_cols if c in df.columns]
            df['UniversalSearch'] = df[existing_cols].astype(str).agg(' '.join, axis=1).str.lower()

            data["trans"] = df
        except Exception as e: st.error(f"Fout Transacties: {e}")

    # B. SYNONIEMEN
    if os.path.exists("Finny_Synonyms.csv"):
        try:
            df = pd.read_csv("Finny_Synonyms.csv", sep=";", dtype=str)
            df['Synoniem'] = df['Synoniem'].str.lower().str.strip()
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            data["syn"] = df
        except: pass

    # C. RGS
    if os.path.exists("Finny_RGS.csv"):
        try:
            df = pd.read_csv("Finny_RGS.csv", sep=";", dtype=str)
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            cols_to_join = [c for c in df.columns if 'Omschrijving' in c or 'Description' in c]
            df['SearchBlob'] = df[cols_to_join].astype(str).agg(' '.join, axis=1).str.lower()
            data["rgs"] = df
        except: pass

    # D. PDF
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages[:15]: text += page.extract_text()
            data["pdf_text"] += f"\n--- BRON: {pdf} ---\n{text[:8000]}"
        except: pass
        
    return data

# 3. LOGICA: INTENT & ANALYSE

def get_intent(client, question):
    """
    Bepaalt Bron, Jaren en Analyse Type.
    """
    context_years = st.session_state["active_years"]
    
    system_prompt = f"""
    Je bent de router van Finny.
    CONTEXT: De gebruiker had het zojuist over de jaren: {context_years}.
    
    TAAK: Analyseer de nieuwe vraag.
    1. 'source': 'PDF' (winst/balans/omzet/solvabiliteit) of 'CSV' (kosten/details/facturen/trends).
    2. 'analysis_type': 
       - 'lookup' (zoek specifiek bedrag/factuur, bv: "kosten vodafone")
       - 'trend' (zoek stijgers/dalers/grootste posten, bv: "welke kosten lopen op?", "grootste kosten", "wat zijn mijn kosten?")
    3. 'years': Welke jaren? ALS GEEN JAAR GENOEMD: Gebruik de jaren uit CONTEXT.
    4. 'keywords': Zoekwoorden. Bijv ["auto"]. Als er gevraagd wordt naar "mijn kosten" (totaal), laat dit leeg.
    
    Output JSON: {{"source": "CSV"|"PDF", "analysis_type": "lookup"|"trend", "years": [2022, 2023], "keywords": ["..."]}}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system", "content": system_prompt}, {"role":"user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        intent = json.loads(res.choices[0].message.content)
        
        # Update geheugen
        if intent.get('years') and len(intent['years']) > 0:
            st.session_state["active_years"] = intent['years']
        else:
            intent['years'] = st.session_state["active_years"]
            
        return intent
    except:
        return {"source": "PDF", "analysis_type": "lookup", "keywords": [], "years": st.session_state["active_years"]}

def analyze_csv_costs(data, intent):
    """
    Slimme CSV Analyse voor Kosten & Trends.
    Houdt rekening met gesloten boekjaren.
    """
    if data["trans"] is None: return "Geen transacties geladen."
    
    df = data["trans"].copy()
    years = intent.get('years', [])
    years = sorted([str(y) for y in years]) # Jaren als strings voor matching
    keywords = intent.get('keywords', [])
    
    if not years:
        return "Geen jaren geselecteerd voor analyse."

    # 1. Filter op Jaren
    df = df[df['Year_Clean'].isin(years)]
    if df.empty:
        return f"Geen data gevonden voor jaren {years}."

    # --- STAP 2: Filteren op 'Echte' Kosten (Gesloten Boekjaar Fix) ---
    
    # A. Verwijder afsluitboekingen op tekst
    # Boekingen met 'Resultaat', 'Winst', 'Balans' in de omschrijving zijn vaak technische boekingen
    mask_closing = df['Description'].astype(str).str.contains(r'(resultaat|winst|balans|afsluiting|verdeling)', case=False, na=False)
    df = df[~mask_closing]

    # B. Filter op Kostenrekeningen (RGS Rubriek 4)
    # Als we geen specifieke keywords hebben (bv vraag "Wat zijn mijn kosten?"), 
    # pakken we alleen GL codes die beginnen met '4' (Kosten in RGS).
    # Als we wel keywords hebben (bv "Auto"), vertrouwen we op de zoekterm en filteren we minder hard op rubriek.
    if not keywords:
        # Algemene kostenvraag -> Alleen Rubriek 4
        df = df[df['Finny_GLCode'].str.startswith('4', na=False)]
    else:
        # Specifieke vraag -> Zoek op keywords in description/categorie
        pattern = '|'.join([re.escape(k.lower()) for k in keywords])
        mask_text = df['UniversalSearch'].str.contains(pattern, na=False)
        df = df[mask_text]

    # 3. Groeperen en Analyseren
    # We groeperen per Categorie (GL Description) en Jaar
    pivot = df.groupby(['Finny_GLDescription', 'Year_Clean'])['AmountDC_num'].sum().unstack(fill_value=0)
    
    # 4. Resultaat Maken
    if pivot.empty:
        return "Geen kosten gevonden na filtering (afsluitboekingen zijn weggelaten)."

    # Sorteer op het laatste jaar
    last_year = years[-1]
    if last_year in pivot.columns:
        pivot = pivot.sort_values(last_year, ascending=False)

    res = f"### KOSTEN ANALYSE ({', '.join(years)})\n"
    res += "*(Afsluitboekingen en resultaatverdelingen zijn uitgezonderd)*\n\n"
    
    # Als we meerdere jaren hebben, bereken verschil
    if len(years) > 1:
        first_year = years[0]
        if first_year in pivot.columns and last_year in pivot.columns:
            pivot['Verschil'] = pivot[last_year] - pivot[first_year]
            # Sorteer op grootste stijgers (grootste positieve verschil)
            pivot = pivot.sort_values('Verschil', ascending=False)
    
    # Toon top 15 rijen
    res += pivot.head(15).to_markdown(floatfmt=".2f")
    
    return res

# 4. UI
if check_password():
    api_key = os.getenv("OPENAI_API_KEY") 
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    
    if not api_key:
        st.error("Geen API Key.")
        st.stop()
        
    client = OpenAI(api_key=api_key)
    data = load_data()

    with st.sidebar:
        # Logo
        logo_files = glob.glob("*.png") + glob.glob("*.jpg")
        if logo_files: st.image(logo_files[0], width=150)
            
        st.title("Finny")
        st.caption(f"Geheugen: {st.session_state['active_years']}")
        if st.button("Nieuw Gesprek"): 
            st.session_state["active_years"] = [2024]
            st.session_state.messages = []
            st.rerun()

    st.title("Finny")
    
    for msg in st.session_state.messages: 
        st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Vraag Finny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("..."):
                # 1. Router
                intent = get_intent(client, prompt)
                
                context = ""
                # 2. Data Ophalen
                if intent['source'] == "PDF":
                    context = data["pdf_text"]
                else:
                    # CSV: Altijd via de slimme kosten-functie
                    # Dit pakt zowel "lookups" als "trends" goed mee met de juiste filters
                    context = analyze_csv_costs(data, intent)
                
                # 3. Antwoord Genereren (TOON AANPASSING)
                
                system_prompt_finny = """
                Je bent Finny, een informele financiële assistent.
                
                BELANGRIJKE STIJLREGELS:
                - Spreek de gebruiker aan met 'je' en 'jij'.
                - GEEN briefformaat! Dus GEEN "Geachte", GEEN "Met vriendelijke groet", GEEN "Finny" aan het eind.
                - Antwoord direct en "to the point".
                - Wees behulpzaam maar niet overdreven beleefd.
                
                INHOUD:
                - Gebruik de tabel in de context voor je antwoord.
                - Als je een trend ziet (stijging/daling), benoem die kort.
                - Verzin geen cijfers.
                """

                # History opbouwen
                messages_payload = [{"role": "system", "content": f"{system_prompt_finny}\n\nDATA CONTEXT:\n{context}"}]
                for msg in st.session_state.messages[-5:]:
                    role = "user" if msg["role"] == "user" else "assistant"
                    messages_payload.append({"role": role, "content": msg["content"]})
                
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_payload
                )
                reply = res.choices[0].message.content
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
