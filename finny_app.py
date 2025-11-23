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
                # Zorg dat jaren integers zijn voor makkelijke vergelijking
                df['Year_Int'] = pd.to_numeric(df['Finny_Year'], errors='coerce').fillna(0).astype(int)
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
    Gebruikt Sessie Geheugen voor context ("die jaren").
    """
    context_years = st.session_state["active_years"]
    
    system_prompt = f"""
    Je bent de router van Finny.
    CONTEXT: De gebruiker had het zojuist over de jaren: {context_years}.
    
    TAAK: Analyseer de nieuwe vraag.
    1. 'source': 'PDF' (omzet/winst/balans/totaal) of 'CSV' (kosten/details/trends).
    2. 'analysis_type': 
       - 'lookup' (zoek specifieke bedragen/facturen, bijv: "wat kostte vodafone?")
       - 'trend' (zoek stijgers/dalers/grootste kosten, bijv: "welke kosten lopen op?", "grootste kostenposten")
    3. 'years': Welke jaren? Als de gebruiker zegt "die jaren" of niks specifieks, gebruik de jaren uit de CONTEXT.
    4. 'keywords': Zoekwoorden (bijv "autokosten"). Bij 'trend' vragen mag dit leeg zijn.
    
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
        
        # Update geheugen als er nieuwe jaren zijn genoemd
        if intent.get('years') and len(intent['years']) > 0:
            st.session_state["active_years"] = intent['years']
        else:
            # Fallback op geheugen als intent leeg is
            intent['years'] = st.session_state["active_years"]
            
        return intent
    except:
        return {"source": "PDF", "analysis_type": "lookup", "keywords": [], "years": st.session_state["active_years"]}

def analyze_csv_trends(data, intent):
    """
    NIEUW: Analyseert trends en stijgers over meerdere jaren.
    Vraag: "Wat zijn de grootste kostenposten die oplopen?"
    """
    if data["trans"] is None: return "Geen transacties geladen."
    
    df = data["trans"].copy()
    years = intent.get('years', [])
    years = sorted([int(y) for y in years if str(y).isdigit()])
    
    if not years:
        return "Geen jaren geselecteerd voor analyse."

    # 1. Filter op jaren
    df = df[df['Year_Int'].isin(years)]
    if df.empty:
        return f"Geen data gevonden voor jaren {years}."

    # 2. Groepeer per Categorie (GL Description) en Jaar
    # We gebruiken Finny_GLDescription als categorie.
    pivot = df.groupby(['Finny_GLDescription', 'Year_Int'])['AmountDC_num'].sum().unstack(fill_value=0)
    
    # 3. Bereken Stijging (Delta)
    # We kijken naar verschil tussen laatste en eerste jaar in de selectie
    first_year = years[0]
    last_year = years[-1]
    
    if first_year == last_year:
        # Als maar 1 jaar: sorteer gewoon op grootte (absoluut)
        pivot['SortValue'] = pivot[first_year].abs()
        explanation = f"Grootste posten in {first_year}:"
    else:
        # Als meerdere jaren: bereken stijging
        pivot['Delta'] = pivot[last_year] - pivot[first_year]
        pivot['SortValue'] = pivot['Delta'] # We sorteren op de stijging (positief = kosten nemen toe, let op: kosten zijn vaak negatief of positief afh van boekhouding. We nemen aan: Kosten = positief in CSV? Of negatief?
        # Aanname: In boekhoudexport zijn kosten vaak positief of negatief.
        # We checken even de data. Als kosten negatief zijn, betekent een 'stijging' dat het getal kleiner wordt (meer min).
        # Voor veiligheid: we kijken naar absolute toename van de 'last'.
        explanation = f"Kostenposten die het hardst zijn gestegen (of veranderd) tussen {first_year} en {last_year}:"

    # 4. Sorteer en Format
    # We sorteren op de absolute verandering om de grootste bewegers te zien
    # Maar de gebruiker vraagt "Oplopen". 
    # Laten we aannemen: Kosten zijn positieve getallen in de vraag-context (AI snapt dat).
    # In de data: check of het Debit/Credit is. Vaak is AmountDC_num - voor credit?
    # We presenteren de ruwe data aan de LLM, die snapt +/- wel.
    
    # We pakken de top 10 grootste bewegers (Delta abs)
    if 'Delta' in pivot.columns:
        top_movers = pivot.reindex(pivot['Delta'].abs().sort_values(ascending=False).index).head(10)
    else:
        top_movers = pivot.sort_values('SortValue', ascending=False).head(10)

    # Verwijder hulpkolommen voor output
    cols_to_drop = ['SortValue', 'Delta'] if 'Delta' in pivot.columns else ['SortValue']
    output_df = top_movers.drop(columns=[c for c in cols_to_drop if c in top_movers.columns])

    res = f"### ANALYSE GROOTSTE KOSTENPOSTEN ({years})\n\n"
    res += f"{explanation}\n\n"
    res += output_df.to_markdown(floatfmt=".2f")
    
    return res

def query_csv_lookup(data, intent):
    """
    De bestaande 'Universal Search' voor specifieke zoekvragen.
    """
    if data["trans"] is None: return "Geen transacties."
    
    df = data["trans"].copy()
    keywords = intent.get('keywords', [])
    years = intent.get('years', [])
    years_str = [str(y) for y in years]

    # 1. Filter Jaar
    if years_str:
        df = df[df['Year_Clean'].isin(years_str)]

    if not keywords:
        return "Geen zoektermen opgegeven voor lookup."

    # 2. Codes Zoeken (Synoniemen/RGS)
    target_codes = set()
    for k in keywords:
        word = k.lower()
        if data["syn"] is not None:
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(word, na=False)]
            if not matches.empty: target_codes.update(matches['Finny_GLCode'].tolist())
        if data["rgs"] is not None:
            matches = data["rgs"][data["rgs"]['SearchBlob'].str.contains(word, na=False)]
            if not matches.empty: target_codes.update(matches['Finny_GLCode'].tolist())

    # 3. Filteren
    mask_code = pd.Series([False] * len(df), index=df.index)
    if target_codes: mask_code = df['Finny_GLCode'].isin(list(target_codes))
    
    pattern = '|'.join([re.escape(k.lower()) for k in keywords])
    mask_text = df['UniversalSearch'].str.contains(pattern, na=False)
    
    df_final = df[mask_code | mask_text]

    # 4. Resultaat
    count = len(df_final)
    total = df_final['AmountDC_num'].sum()
    
    if count == 0: return f"Geen data gevonden voor '{keywords}' in {years}."
    
    res = f"Gevonden: {count} transacties.\nTotaal: € {total:,.2f}\n"
    
    cols = ['EntryDate', 'Description', 'AccountName', 'Finny_GLDescription', 'AmountDC_num']
    valid_cols = [c for c in cols if c in df_final.columns]
    
    limit = 50
    res += f"\nLijst (max {limit}):\n{df_final[valid_cols].head(limit).to_string(index=False)}"
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
        # Logo (Zoekt elk plaatje)
        logo_files = glob.glob("*.png") + glob.glob("*.jpg")
        if logo_files: st.image(logo_files[0], width=150)
            
        st.title("🦁 Finny")
        st.caption(f"Actieve Jaren: {st.session_state['active_years']}")
        if st.button("Reset Geheugen"): 
            st.session_state["active_years"] = [2024]
            st.session_state.messages = []
            st.rerun()

    st.title("Finny Demo")
    
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
                    st.caption(f"Bron: PDF | Jaren: {intent['years']}")
                else:
                    # CSV KEUZE: TREND of LOOKUP?
                    if intent['analysis_type'] == 'trend':
                        st.caption(f"Bron: CSV (Trend Analyse) | Jaren: {intent['years']}")
                        context = analyze_csv_trends(data, intent)
                    else:
                        st.caption(f"Bron: CSV (Zoeken) | Zoekt: {intent['keywords']} | Jaren: {intent['years']}")
                        context = query_csv_lookup(data, intent)
                
                # 3. Antwoord Genereren (Met Geschiedenis!)
                # We bouwen de history op voor de LLM
                messages_payload = [
                    {"role": "system", "content": f"""Je bent Finny. 
                    Gebruik deze DATA CONTEXT voor je antwoord:\n{context}\n
                    INSTRUCTIES:
                    - Als je een tabel met trends ziet, benoem de grootste stijgers expliciet met bedragen.
                    - Reken niet zelf als totalen er al staan.
                    - Wees zakelijk en direct."""}
                ]
                
                # Voeg laatste 4 berichten toe voor context
                for msg in st.session_state.messages[-5:]:
                    # Map streamlit roles naar openai roles
                    role = "user" if msg["role"] == "user" else "assistant"
                    messages_payload.append({"role": role, "content": msg["content"]})
                
                # Voeg huidige vraag toe (als die er nog niet in staat, maar we stuurden net history)
                # History bevat de huidige vraag al door de append bovenaan? Nee, messages[-5:] pakt hem mee.
                
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_payload
                )
                reply = res.choices[0].message.content
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
