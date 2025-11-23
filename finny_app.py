import streamlit as st
import pandas as pd
import os
import json
import re
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# 1. CONFIGURATIE
st.set_page_config(page_title="Finny", layout="wide")
load_dotenv()

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="pw", on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# 2. DATA LADEN (VERIFIEERDE METHODE)
@st.cache_data
def load_data():
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": ""}
    
    # A. Transacties
    if os.path.exists("Finny_Transactions.csv"):
        try:
            # Lees CSV met puntkomma
            df = pd.read_csv("Finny_Transactions.csv", sep=";", dtype=str)
            
            # Schoonmaak 1: Bedragen (1.000,00 -> 1000.00)
            if 'AmountDC_num' in df.columns:
                df['AmountDC_num'] = (
                    df['AmountDC_num']
                    .astype(str)
                    .str.replace(',', '.')
                    .replace('nan', '0')
                )
                df['AmountDC_num'] = pd.to_numeric(df['AmountDC_num'], errors='coerce').fillna(0.0)
            
            # Schoonmaak 2: Codes (2600.0 -> 2600)
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].astype(str).str.split('.').str[0].str.strip()
                
            data["trans"] = df
        except Exception as e:
            st.error(f"Fout Transacties: {e}")

    # B. Synoniemen
    if os.path.exists("Finny_Synonyms.csv"):
        try:
            df = pd.read_csv("Finny_Synonyms.csv", sep=";", dtype=str)
            df['Synoniem'] = df['Synoniem'].str.lower().str.strip()
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].astype(str).str.split('.').str[0].str.strip()
            data["syn"] = df
        except: pass

    # C. RGS
    if os.path.exists("Finny_RGS.csv"):
        try:
            df = pd.read_csv("Finny_RGS.csv", sep=";", dtype=str)
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].astype(str).str.split('.').str[0].str.strip()
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

# 3. LOGICA (STAPPENPLAN)

def get_intent(client, question):
    """Bepaalt of we naar PDF of CSV moeten."""
    system_prompt = """
    Je bent de router.
    1. 'PDF' -> Vragen over jaarrekening, balans, winst, strategie.
    2. 'CSV' -> Vragen over specifieke kosten, leveranciers, bedragen, details.
    
    Output JSON: {"source": "CSV"|"PDF", "years": [2023], "keywords": ["zoekterm"]}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system", "content": system_prompt}, {"role":"user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"source": "PDF", "keywords": [], "years": []}

def query_csv_exact(data, intent):
    """
    Voert de zoektocht uit op basis van jouw stappenplan.
    """
    if data["trans"] is None: return "Geen transacties."
    
    df = data["trans"].copy()
    keywords = intent.get('keywords', [])
    years = intent.get('years', [])
    debug_info = []

    # 1. Filter op Jaar (Optioneel)
    if years and 'Finny_Year' in df.columns:
        df['Year_Clean'] = df['Finny_Year'].astype(str).str.split('.').str[0]
        years_str = [str(y) for y in years]
        df_year = df[df['Year_Clean'].isin(years_str)]
        if not df_year.empty:
            df = df_year
            debug_info.append(f"Gefilterd op jaren: {years_str}")

    if not keywords:
        return "Geen zoektermen.", []

    # 2. Verzamel GL Codes (Uit Synoniemen & RGS)
    target_codes = set()
    
    for k in keywords:
        word = k.lower()
        # Check Synoniemen
        if data["syn"] is not None:
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(word, na=False)]
            if not matches.empty:
                codes = matches['Finny_GLCode'].tolist()
                target_codes.update(codes)
                debug_info.append(f"Synoniem '{word}' -> Codes: {codes}")

        # Check RGS
        if data["rgs"] is not None:
            matches = data["rgs"][data["rgs"]['RGS_Omschrijving'].str.lower().str.contains(word, na=False)]
            if not matches.empty:
                codes = matches['Finny_GLCode'].tolist()
                target_codes.update(codes)
                debug_info.append(f"RGS '{word}' -> Codes: {codes}")

    # 3. Filteren (Code OF Tekst)
    if target_codes:
        # Filter op gevonden codes
        mask = df['Finny_GLCode'].isin(list(target_codes))
        df_final = df[mask]
    else:
        # Geen codes? Filter op tekst in Omschrijving
        pattern = '|'.join([re.escape(k) for k in keywords])
        mask = df['Description'].astype(str).str.contains(pattern, case=False, na=False)
        df_final = df[mask]
        debug_info.append(f"Geen codes, gezocht op tekst '{pattern}'")

    # 4. Resultaat
    total = df_final['AmountDC_num'].sum()
    count = len(df_final)
    
    res = f"Aantal transacties: {count}\nTotaal: € {total:,.2f}\n"
    
    cols = ['EntryDate', 'Description', 'AmountDC_num', 'Finny_GLCode']
    valid_cols = [c for c in cols if c in df_final.columns]
    
    if count < 50:
        res += f"\n{df_final[valid_cols].to_string(index=False)}"
    else:
        try:
            top = df_final.groupby('Description')['AmountDC_num'].sum().sort_values().head(5)
            res += f"\nTop 5 Kostenposten:\n{top.to_string()}"
        except: pass
            
    return res, debug_info

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
        st.title("Finny")
        if data["trans"] is not None: st.success(f"Transacties: {len(data['trans'])}")
        if st.button("Reset"): st.rerun()

    st.title("Finny Demo")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Vraag Finny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("..."):
                intent = get_intent(client, prompt)
                
                context = ""
                if intent['source'] == "PDF":
                    context = data["pdf_text"]
                    st.caption(f"Bron: PDF | Zoekt: {intent['keywords']}")
                else:
                    res_text, debug_log = query_csv_exact(data, intent)
                    context = res_text
                    # Toon hoe we gezocht hebben (voor controle)
                    with st.expander("Details zoekopdracht"):
                        st.write(f"**Strategie:** {intent['source']} | **Jaren:** {intent['years']}")
                        for step in debug_log: st.write(step)
                
                sys_msg = "Je bent Finny. Gebruik de context. Reken niet zelf, totalen zijn al berekend."
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system", "content": sys_msg + f"\nCONTEXT:\n{context}"},
                        {"role":"user", "content": prompt}
                    ]
                )
                reply = res.choices[0].message.content
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
