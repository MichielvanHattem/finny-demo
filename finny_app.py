import streamlit as st
import pandas as pd
import os
import json
import re
import glob
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# 1. CONFIGURATIE
st.set_page_config(page_title="Finny", layout="wide")
load_dotenv()

# Geheugen
if "active_years" not in st.session_state:
    st.session_state["active_years"] = [2024]
if "messages" not in st.session_state: 
    st.session_state.messages = []

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="pw", on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# 2. DATA LADEN
@st.cache_data(ttl=3600)
def load_data():
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": "", "latest_year": 2024}
    
    def clean_code(val):
        return str(val).split('.')[0].strip()

    # A. TRANSACTIES
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", dtype=str)
            if 'AmountDC_num' in df.columns:
                df['AmountDC_num'] = df['AmountDC_num'].str.replace(',', '.').replace('nan', '0')
                df['AmountDC_num'] = pd.to_numeric(df['AmountDC_num'], errors='coerce').fillna(0.0)
            
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            
            if 'Finny_Year' in df.columns:
                df['Year_Clean'] = df['Finny_Year'].astype(str).str.split('.').str[0]
                valid_years = pd.to_numeric(df['Year_Clean'], errors='coerce').dropna()
                if not valid_years.empty:
                    data["latest_year"] = int(valid_years.max())

            # Universal Search (Alles doorzoekbaar maken)
            cols = ['Description', 'AccountName', 'Finny_GLDescription', 'Finny_GLCode']
            existing = [c for c in cols if c in df.columns]
            df['UniversalSearch'] = df[existing].astype(str).agg(' '.join, axis=1).str.lower()

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
            cols = [c for c in df.columns if 'Omschrijving' in c or 'Description' in c]
            df['SearchBlob'] = df[cols].astype(str).agg(' '.join, axis=1).str.lower()
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

# 3. LOGICA

def get_intent(client, question):
    """
    Router: Bepaalt Bron (PDF/CSV) en Context (Jaren).
    """
    context_years = st.session_state["active_years"]
    if not context_years: context_years = [2024]
    
    system_prompt = f"""
    Je bent de router van Finny.
    CONTEXT: Huidige focusjaren: {context_years}.
    
    TAAK: Bepaal de bron.
    1. 'PDF' -> ALLEEN voor: Totale Winst, Totale Omzet, Balanstotaal, Solvabiliteit.
    2. 'CSV' -> ALTIJD voor: Specifieke kosten (Telefoon, Auto, Huisvesting, Kantoor), leveranciers, detailvragen.
       (Zelfs als de gebruiker vraagt "Wat zijn de totale telefoonkosten?", is dit CSV).
    
    Output JSON: {{"source": "CSV"|"PDF", "years": [2022, 2023, 2024], "keywords": ["zoekterm"]}}
    ALS GEEN JAAR GENOEMD: Gebruik de jaren uit de CONTEXT.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system", "content": system_prompt}, {"role":"user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        intent = json.loads(res.choices[0].message.content)
        
        if intent.get('years'):
            st.session_state["active_years"] = intent['years']
        else:
            intent['years'] = context_years
            
        return intent
    except:
        return {"source": "PDF", "keywords": [], "years": context_years}

def expand_search_terms(keywords):
    """
    Slimme Synoniemen Booster.
    Als de gebruiker 'telefoon' zegt, zoeken we ook op 'communicatie'.
    """
    expanded = set(keywords)
    mapping = {
        'telefoon': ['communicatie', 'mobiel', 'internet', 'bellen', 'kpn', 'vodafone', 't-mobile', 'ziggo'],
        'communicatie': ['telefoon', 'internet', 'data', 'mobiel'],
        'auto': ['vervoer', 'brandstof', 'lease', 'parkeren', 'tank', 'kilometer'],
        'kantoor': ['huisvesting', 'huur', 'inventaris', 'bureau', 'supplies'],
        'personeel': ['loon', 'salaris', 'sociale', 'verzekering']
    }
    
    for k in keywords:
        for key, values in mapping.items():
            if key in k.lower():
                expanded.update(values)
    
    return list(expanded)

def analyze_csv_specific(data, intent):
    """
    Specifieke Kosten Analyse (CSV).
    Maakt een harde tabel per jaar voor de gevraagde kostenpost.
    """
    if data["trans"] is None: return "Geen transacties."
    
    df = data["trans"].copy()
    years = intent.get('years', [])
    # Zorg dat jaren strings zijn voor matching
    years = sorted([str(y) for y in years])
    
    raw_keywords = intent.get('keywords', [])
    # Breid zoektermen uit (Telefoon -> Communicatie)
    keywords = expand_search_terms(raw_keywords)
    
    # 1. Filter Jaren
    df = df[df['Year_Clean'].isin(years)]
    if df.empty: return f"Geen data voor jaren {years}."

    # 2. Filter op 'Echte' Kosten (Geen resultaatboekingen)
    mask_tech = df['Description'].astype(str).str.contains(r'(resultaat|winst|balans|afsluiting)', case=False, na=False)
    df = df[~mask_tech]

    # 3. Filter op Zoektermen (Universal Search)
    if keywords:
        pattern = '|'.join([re.escape(k.lower()) for k in keywords])
        # We zoeken ook in de RGS-omschrijvingen via de koppeling, maar UniversalSearch bevat GLDescription al.
        mask_text = df['UniversalSearch'].str.contains(pattern, na=False)
        
        # Extra check: Filter ook op GL Codes die in RGS/Synoniemen matchen
        target_codes = []
        if data["rgs"] is not None:
            rgs_hits = data["rgs"][data["rgs"]['SearchBlob'].str.contains(pattern, na=False)]
            target_codes.extend(rgs_hits['Finny_GLCode'].tolist())
            
        if target_codes:
            mask_code = df['Finny_GLCode'].isin(target_codes)
            df = df[mask_text | mask_code]
        else:
            df = df[mask_text]

    if df.empty:
        return f"Ik zie geen transacties voor '{raw_keywords}' in {years}."

    # 4. Groeperen per Jaar (Harde Cijfers)
    # We sommeren alles wat we gevonden hebben per jaar
    pivot = df.groupby('Year_Clean')['AmountDC_num'].sum().reset_index()
    pivot.columns = ['Jaar', 'Bedrag']
    
    # Zorg dat alle gevraagde jaren erin staan (ook als bedrag 0 is)
    for y in years:
        if y not in pivot['Jaar'].values:
            pivot = pd.concat([pivot, pd.DataFrame({'Jaar': [y], 'Bedrag': [0.0]})])
            
    pivot = pivot.sort_values('Jaar')

    # 5. Output Maken
    res = f"### KOSTEN OVERZICHT: {', '.join(raw_keywords).upper()}\n\n"
    
    # Tabel genereren
    res += "| Jaar | Totaal Bedrag |\n|---|---|\n"
    for _, row in pivot.iterrows():
        res += f"| {row['Jaar']} | € {row['Bedrag']:,.2f} |\n"
    
    # Grootste posten tonen (Details)
    res += "\n**Grootste boekingen/posten in deze selectie:**\n"
    top_items = df.groupby(['Description', 'Year_Clean'])['AmountDC_num'].sum().reset_index().sort_values('AmountDC_num', ascending=False).head(5)
    res += top_items.to_markdown(index=False, floatfmt=".2f")

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
        logo_files = glob.glob("*.png") + glob.glob("*.jpg")
        if logo_files: st.image(logo_files[0], width=150)
            
        st.title("Finny")
        
        if data["trans"] is not None:
            if not st.session_state["active_years"]:
                st.session_state["active_years"] = [str(data["latest_year"])]
            st.caption(f"Geheugen: {st.session_state['active_years']}")
        
        if st.button("Nieuw Gesprek"): 
            st.session_state["active_years"] = [str(data["latest_year"])]
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
                intent = get_intent(client, prompt)
                
                context = ""
                if intent['source'] == "PDF":
                    context = data["pdf_text"]
                    st.caption(f"Bron: Jaarrekening | Jaren: {intent['years']}")
                else:
                    # CSV Analyse
                    context = analyze_csv_specific(data, intent)
                    st.caption(f"Bron: Transacties | Focus: {intent['keywords']} | Jaren: {intent['years']}")
                
                system_prompt_finny = """
                Je bent Finny, een informele financiële assistent.
                STIJL:
                - Spreek aan met 'je/jij'.
                - GEEN briefformaat (geen "Geachte", geen "Groet").
                - Direct en zakelijk.
                
                INSTRUCTIES:
                - Gebruik de TABEL uit de context.
                - Als er voor een jaar € 0.00 staat, zeg dan dat er geen kosten waren.
                - Verzin geen bedragen.
                - Als je een stijging/daling ziet in de tabel, benoem die.
                """
                
                # History meesturen
                messages_payload = [{"role": "system", "content": f"{system_prompt_finny}\n\nDATA:\n{context}"}]
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
