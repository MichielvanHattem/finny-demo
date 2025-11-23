import streamlit as st
import pandas as pd
import os
import json
import re
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# 1. CONFIGURATIE
st.set_page_config(page_title="Finny | Financial Assistant", page_icon="🦁", layout="wide")
load_dotenv() # Laad .env voor lokaal gebruik

# Wachtwoord beveiliging
def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="pw", on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# 2. DATA LADEN (De 3 Bronnen)
@st.cache_data
def load_data():
    data = {"rgs": None, "syn": None, "trans": None, "pdf_text": ""}
    
    # A. CSV's LADEN (Met puntkomma ; gescheiden)
    try:
        if os.path.exists("Finny_RGS.csv"):
            data["rgs"] = pd.read_csv("Finny_RGS.csv", sep=";")
        if os.path.exists("Finny_Synonyms.csv"):
            data["syn"] = pd.read_csv("Finny_Synonyms.csv", sep=";")
            # Zorg dat synoniemen lowercase zijn voor makkelijk zoeken
            data["syn"]['Synoniem'] = data["syn"]['Synoniem'].str.lower()
        if os.path.exists("Finny_Transactions.csv"):
            data["trans"] = pd.read_csv("Finny_Transactions.csv", sep=";")
    except Exception as e:
        st.error(f"Fout bij laden CSV's: {e}")

    # B. PDF's LADEN (Zoek alle PDF's in de map)
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            # Scan eerste 20 pagina's (meestal genoeg voor balans/winst)
            for page in reader.pages[:20]: 
                text += page.extract_text()
            data["pdf_text"] += f"\n--- BRON: {pdf} ---\n{text[:10000]}"
        except: pass
        
    return data

# 3. LOGICA (ROUTER & ENGINES)

def get_intent(client, question):
    """Bepaalt of we naar PDF (Jaarverslag) of CSV (Transacties) moeten."""
    system_prompt = """
    Je bent de router van Finny.
    Bepaal de bron voor de vraag:
    1. 'PDF' -> Vragen over omzet, winst, balans, solvabiliteit, jaarcijfers, strategie.
    2. 'CSV' -> Vragen over specifieke kosten, leveranciers (KPN, Shell), bedragen, trends per maand.
    
    Geef ook zoekwoorden en jaren mee.
    Antwoord JSON: {"source": "CSV"|"PDF", "years": [2023], "keywords": ["term1"]}
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
        return {"source": "PDF", "keywords": [], "years": []} # Fallback

def query_csv_smart(data, intent):
    """De stappen voor CSV: Synoniemen -> Filteren -> Rekenen"""
    if data["trans"] is None: return "Geen transacties geladen."
    
    df = data["trans"].copy()
    keywords = intent.get('keywords', [])
    years = intent.get('years', [])
    
    # STAP 1: Filter op Jaar
    if years and 'Finny_Year' in df.columns:
        df_year = df[df['Finny_Year'].isin(years)]
        if not df_year.empty:
            df = df_year # Alleen filteren als we data overhouden
            
    # STAP 2: Synoniemen Check
    gl_codes = []
    if data["syn"] is not None and keywords:
        for word in keywords:
            # Zoek exact of deel match in synoniemenlijst
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(word.lower(), na=False)]
            if not matches.empty:
                gl_codes.extend(matches['Finny_GLCode'].tolist())
    
    # STAP 3: Filter Transacties
    if gl_codes:
        # Filter op GL Code (heel precies)
        mask = df['Finny_GLCode'].isin(gl_codes)
        df_filtered = df[mask]
    elif keywords:
        # Geen synoniem? Zoek dan in de omschrijving (Description)
        pattern = '|'.join([re.escape(k) for k in keywords])
        mask = df['Description'].astype(str).str.contains(pattern, case=False, na=False)
        df_filtered = df[mask]
    else:
        df_filtered = df # Geen filters? Alles tonen (of niks, afhankelijk van voorkeur)

    # STAP 4: Aggregeren & Resultaat
    total = df_filtered['AmountDC_num'].sum()
    count = len(df_filtered)
    
    if count == 0:
        return f"Geen transacties gevonden voor {keywords}."
    
    # Tekst voor de LLM
    context = f"Gevonden transacties: {count}\nTotaal bedrag: € {total:,.2f}\n"
    if count < 30:
        context += f"\nDetails:\n{df_filtered[['EntryDate', 'Description', 'AmountDC_num']].to_string(index=False)}"
    else:
        # Top 10 kostenposten
        top = df_filtered.groupby('Description')['AmountDC_num'].sum().sort_values().head(10)
        context += f"\nTop 10 Posten:\n{top.to_string()}"
        
    return context

# 4. DE APP
if check_password():
    # API Key Check
    api_key = os.getenv("OPENAI_API_KEY") 
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    
    if not api_key:
        st.error("Geen API Key gevonden. Zet deze in .env of secrets.")
        st.stop()
        
    client = OpenAI(api_key=api_key)
    data = load_data()

    # --- SIDEBAR ---
    with st.sidebar:
        # Logo check (jpg/png)
        logo_found = False
        for ext in ["jpg", "png", "jpeg"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                logo_found = True
                break
        if not logo_found:
            st.title("🦁 Finny")
            
        st.markdown("### Status")
        if data["trans"] is not None: st.success("✅ Transacties geladen")
        if data["rgs"] is not None: st.success("✅ RGS Schema geladen")
        if data["pdf_text"]: st.success("✅ PDF's geladen")
        
        if st.button("Reset"): st.rerun()

    # --- HOOFDSCHERM ---
    st.title("Finny Demo")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: 
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    if prompt := st.chat_input("Vraag Finny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Even nadenken..."):
                # 1. Router
                intent = get_intent(client, prompt)
                st.caption(f"Strategie: {intent['source']} | Zoektermen: {intent['keywords']}")
                
                # 2. Data ophalen
                context = ""
                
                # Eerst PDF checken als dat de strategie is
                if intent['source'] == "PDF":
                    context = data["pdf_text"]
                else:
                    # Anders CSV stappen doorlopen
                    context = query_csv_smart(data, intent)
                
                # 3. Antwoord
                system_msg = "Je bent Finny. Gebruik de context. Reken niet zelf, totalen zijn al berekend."
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system", "content": system_msg + f"\nCONTEXT:\n{context}"},
                        {"role":"user", "content": prompt}
                    ]
                )
                reply = res.choices[0].message.content
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
