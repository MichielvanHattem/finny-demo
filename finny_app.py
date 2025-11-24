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

# Geheugen initialisatie (Leeg starten, wordt gevuld door data)
if "active_years" not in st.session_state:
    st.session_state["active_years"] = []
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
    # We slaan nu 'all_years' op in plaats van alleen 'latest_year'
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": "", "all_years": [2024]}
    
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
                
                # HIER HALEN WE ALLE JAREN OP
                valid_years = pd.to_numeric(df['Year_Clean'], errors='coerce').dropna().unique()
                if len(valid_years) > 0:
                    # Sorteer de jaren (2022, 2023, 2024)
                    data["all_years"] = sorted(valid_years.astype(int).tolist())

            # Universal Search
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
    Router: Bepaalt Bron & Context.
    """
    context_years = st.session_state["active_years"]
    # Als context leeg is (eerste keer), gebruik data-defaults (wordt later gezet)
    
    system_prompt = f"""
    Je bent de router van Finny.
    CONTEXT: Huidige focusjaren: {context_years}.
    
    TAAK:
    1. 'source': 'PDF' (winst/balans/omzet) of 'CSV' (kosten/details/leveranciers).
    2. 'analysis_type': 'lookup' (bedrag zoeken) of 'trend' (verloop).
    3. 'years': Welke jaren? ALS GEEN JAAR GENOEMD: Gebruik de CONTEXT jaren.
    4. 'keywords': Zoekwoorden.
    
    Output JSON.
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
        return {"source": "PDF", "analysis_type": "lookup", "keywords": [], "years": context_years}

def analyze_csv_costs(data, intent):
    """
    Analyseert kosten en pakt de Categorie uit Synoniemen.
    """
    if data["trans"] is None: return "Geen transacties."
    
    df = data["trans"].copy()
    years = intent.get('years', [])
    years = sorted([str(y) for y in years])
    keywords = intent.get('keywords', [])
    
    # 1. Filter Jaren
    df = df[df['Year_Clean'].isin(years)]
    if df.empty: return f"Geen data voor jaren {years}."

    # 2. Filter 'Echte' Kosten
    mask_tech = df['Description'].astype(str).str.contains(r'(resultaat|winst|balans|afsluiting)', case=False, na=False)
    df = df[~mask_tech]

    # 3. BEPAAL CATEGORIE UIT SYNONIEMEN
    found_categories = set()
    if keywords and data["syn"] is not None:
        for k in keywords:
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(k.lower(), na=False)]
            if not matches.empty and 'Categorie' in matches.columns:
                found_categories.update(matches['Categorie'].unique().tolist())

    # 4. BEREKENINGEN

    # A. Specifieke Zoekopdracht
    df_specific = pd.DataFrame()
    if keywords:
        pattern = '|'.join([re.escape(k.lower()) for k in keywords])
        mask_spec = df['UniversalSearch'].str.contains(pattern, na=False)
        df_specific = df[mask_spec]
    
    # B. Categorie Totaal
    df_category = pd.DataFrame()
    if found_categories:
        cat_pattern = '|'.join([re.escape(c) for c in found_categories])
        mask_cat = df['Finny_GLDescription'].astype(str).str.contains(cat_pattern, case=False, na=False)
        df_category = df[mask_cat]
    else:
        if not keywords:
             df_category = df[df['Finny_GLCode'].str.startswith('4', na=False)]

    # 5. Output
    res = f"### ANALYSE ({', '.join(years)})\n"
    
    if not df_specific.empty:
        pivot_spec = df_specific.groupby('Year_Clean')['AmountDC_num'].sum().reset_index()
        pivot_spec.columns = ['Jaar', f'Totaal "{", ".join(keywords)}"',]
        res += f"\n**Zoekterm '{', '.join(keywords)}':**\n"
        res += pivot_spec.to_markdown(index=False, floatfmt=".2f") + "\n\n"
    elif keywords:
        res += f"Geen specifieke transacties voor '{keywords}'.\n\n"

    if not df_category.empty and found_categories:
        cat_names = ", ".join(found_categories)
        pivot_cat = df_category.groupby('Year_Clean')['AmountDC_num'].sum().reset_index()
        pivot_cat.columns = ['Jaar', f'Totaal Categorie: {cat_names}']
        res += f"\n**Vergelijking Categorie '{cat_names}':**\n"
        res += pivot_cat.to_markdown(index=False, floatfmt=".2f") + "\n"
        
        top = df_category.groupby('Description')['AmountDC_num'].sum().sort_values(ascending=False).head(5).reset_index()
        res += f"\n*Grootste posten in deze categorie:*\n"
        res += top.to_markdown(index=False, floatfmt=".2f")

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
            # ZET DEFAULT JAREN OP ALLE BESCHIKBARE JAREN (2022, 2023, 2024)
            if not st.session_state["active_years"]:
                # Convert int list to string list for display/logic
                st.session_state["active_years"] = [str(y) for y in data["all_years"]]
            
            st.caption(f"Geheugen: {st.session_state['active_years']}")
        
        if st.button("Nieuw Gesprek"): 
            # Reset naar ALLE jaren
            st.session_state["active_years"] = [str(y) for y in data["all_years"]]
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
                    context = analyze_csv_costs(data, intent)
                    st.caption(f"Bron: Transacties | Focus: {intent['keywords']}")
                
                system_prompt_finny = """
                Je bent Finny, een informele financiële assistent.
                STIJL:
                - Spreek aan met 'je/jij'.
                - GEEN briefformaat.
                - Direct en zakelijk.
                
                INSTRUCTIES:
                - Gebruik de TABELLEN uit de context.
                - Als je specifieke kosten toont én een categorie-totaal, leg het verschil uit.
                """
                
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
