import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
import re
from datetime import datetime

# ==========================================
# 1. CONFIGURATIE
# ==========================================
st.set_page_config(page_title="Finny | Intelligent Finance", page_icon="💰", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="password_input", on_change=lambda: st.session_state.update({"password_correct": st.session_state.password_input == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# ==========================================
# 2. DATA LADEN & LIVE SCHOONMAKEN
# ==========================================
@st.cache_data
def load_data():
    data = {"transacties": None, "pdf_text": "", "syllabus": ""}
    
    # --- A. CSV (VUIILE DATA INLEZEN EN SCHOONMAKEN) ---
    # We zoeken naar jouw originele bestand
    csv_file = "133700 FinTransactionSearch all 5jr.csv"
    
    if os.path.exists(csv_file):
        try:
            # 1. Lees het 'vuile' Twinfield bestand
            df = pd.read_csv(csv_file, sep=";", on_bad_lines='skip', low_memory=False)
            
            # 2. Kolommen normaliseren
            df.columns = df.columns.str.strip().str.lower()
            
            # 3. Hernoemen naar standaard namen
            rename_map = {}
            for col in df.columns:
                if 'bedrag' in col or 'value' in col or 'amount' in col: rename_map[col] = 'bedrag'
                if 'datum' in col or 'date' in col: rename_map[col] = 'datum'
                if 'omschrijving' in col or 'desc' in col: rename_map[col] = 'omschrijving'
                if 'grootboek' in col or 'dim1' in col: rename_map[col] = 'grootboek'
            
            df = df.rename(columns=rename_map)
            
            # 4. Bedragen repareren (De 'Dutch Number Fix')
            if 'bedrag' in df.columns:
                def clean_money(val):
                    if pd.isna(val): return 0.0
                    # Verwijder alles behalve cijfers, min-teken en komma
                    s = str(val).replace('€', '').replace(' ', '').replace('.', '') 
                    s = s.replace(',', '.') # Komma naar punt
                    try: return float(s)
                    except: return 0.0
                df['clean_amount'] = df['bedrag'].apply(clean_money)
            
            # 5. Datums repareren
            if 'datum' in df.columns:
                df['clean_date'] = pd.to_datetime(df['datum'], dayfirst=True, errors='coerce')
                df['year'] = df['clean_date'].dt.year
            
            # 6. Zoektekst maken
            cols = [c for c in ['omschrijving', 'grootboek'] if c in df.columns]
            df['search_text'] = df[cols].astype(str).agg(' '.join, axis=1).str.lower()
            
            data["transacties"] = df
            
        except Exception as e:
            st.error(f"CSV Fout: {e}")
    
    # --- B. PDF LADEN ---
    pdfs = ["Van Hattem Advies B.V. - Jaarrekening 2024.pdf", "Van Hattem Advies B.V. - Jaarrekening 2023.pdf", "Van Hattem Advies B.V. - Jaarstukken 2022.pdf"]
    for pdf in pdfs:
        if os.path.exists(pdf):
            try:
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages[:20]: text += page.extract_text()
                data["pdf_text"] += f"--- BRON: {pdf} ---\n{text[:10000]}\n\n"
            except: pass
            
    if os.path.exists("Finny_syllabus.txt"):
        with open("Finny_syllabus.txt", "r", encoding="utf-8") as f: data["syllabus"] = f.read()

    return data

# ==========================================
# 3. LOGICA (ROUTER & ENGINE)
# ==========================================
def run_router(client, question, context_snippet):
    system_prompt = f"""
    Je bent de Router.
    CONTEXT: {context_snippet[:2000]}
    VRAAG: {question}
    
    TAAK:
    1. Bepaal bron: 'CSV' (details/kosten/leveranciers) of 'PDF' (omzet/winst/balans).
    2. Vertaal zoektermen: "Auto" -> ["brandstof", "lease", "garage"].
    
    ANTWOORD JSON: {{"source": "CSV"|"PDF"|"BOTH", "years": [2023], "keywords": ["term1"]}}
    """
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":system_prompt}], response_format={"type":"json_object"})
        return json.loads(res.choices[0].message.content)
    except: return {"source": "BOTH", "years": [], "keywords": []}

def query_csv(df, intent):
    if df is None: return "Geen CSV data beschikbaar."
    
    filtered = df.copy()
    
    # Filter Jaren
    if intent.get('years') and 'year' in df.columns:
        # Alleen filteren als het resultaten oplevert
        year_match = filtered[filtered['year'].isin(intent['years'])]
        if len(year_match) > 0: filtered = year_match
    
    # Filter Keywords
    keywords = intent.get('keywords', [])
    if keywords:
        pattern = '|'.join([re.escape(k.lower()) for k in keywords])
        filtered = filtered[filtered['search_text'].str.contains(pattern, na=False)]
        
    count = len(filtered)
    if count == 0: return f"Geen transacties gevonden voor {keywords}."
    
    total = filtered['clean_amount'].sum()
    
    if count > 50:
        # Top 10 kostenposten tonen
        top = filtered.groupby('omschrijving')['clean_amount'].sum().sort_values().head(10)
        return f"--- CSV SAMENVATTING ---\nAantal: {count}\nTOTAAL: € {total:,.2f}\n\nTop posten:\n{top.to_string()}"
    else:
        cols = [c for c in ['datum', 'omschrijving', 'bedrag'] if c in filtered.columns]
        return f"--- CSV DETAILS ---\nTOTAAL: € {total:,.2f}\n\n{filtered[cols].to_string(index=False)}"

# ==========================================
# 4. APP UI
# ==========================================
if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt.")
        st.stop()

    data = load_data()
    
    with st.sidebar:
        st.title("🤖 Finny")
        if data["transacties"] is not None: st.success(f"✅ Data Live ({len(data['transacties'])} regels)")
        else: st.error("❌ Geen CSV gevonden")
        if st.button("Reset"): st.rerun()

    st.title("Finny 7.5 (Self-Cleaning)")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Denken..."):
                intent = run_router(client, prompt, data["syllabus"])
                st.caption(f"Strategie: {intent}")
                
                context = ""
                if intent.get("source") in ["PDF", "BOTH"]: context += data["pdf_text"]
                if intent.get("source") in ["CSV", "BOTH"]: context += query_csv(data["transacties"], intent)
                
                sys_msg = "Je bent Finny. Gebruik de context. Reken niet zelf, de totalen staan er al."
                res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system", "content": sys_msg + context}, {"role":"user", "content": prompt}])
                
                reply = res.choices[0].message.content
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
