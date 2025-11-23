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
# Icoon aangepast naar zakelijke vrouw
st.set_page_config(page_title="Finny", page_icon="👩‍💼", layout="wide")
load_dotenv()

# Geheugen initialisatie (Default leeg, wordt gevuld door data)
if "active_years" not in st.session_state:
    st.session_state["active_years"] = []

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="pw", on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# 2. DATA LADEN
@st.cache_data
def load_data():
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": "", "years_found": []}
    
    def clean_code(val):
        return str(val).split('.')[0].strip()

    # A. TRANSACTIES
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", dtype=str)
            
            # Bedragen
            if 'AmountDC_num' in df.columns:
                df['AmountDC_num'] = df['AmountDC_num'].str.replace(',', '.').replace('nan', '0')
                df['AmountDC_num'] = pd.to_numeric(df['AmountDC_num'], errors='coerce').fillna(0.0)
            
            # Codes
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            
            # Jaren (Cruciaal voor jouw vraag!)
            if 'Finny_Year' in df.columns:
                df['Year_Clean'] = df['Finny_Year'].astype(str).str.split('.').str[0]
                # Sla op welke jaren we hebben gevonden voor de sidebar
                data["years_found"] = sorted(df['Year_Clean'].unique().tolist())

            # Universal Search kolom
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

# 3. LOGICA

def get_intent(client, question):
    """Router en Geheugen."""
    # Pak context uit sessie (wat was het laatste jaar waar we over praatten?)
    active_years = st.session_state["active_years"]
    
    system_prompt = f"""
    Je bent de router.
    CONTEXT: De gebruiker had het eerder over deze jaren: {active_years}.
    
    TAAK:
    1. 'source': 'PDF' (jaarrekening/winst/balans) of 'CSV' (kosten/details/leveranciers).
    2. 'keywords': zelfstandige naamwoorden (bijv: "kantoorkosten", "auto").
    3. 'years': Jaren in de vraag. ALS GEEN JAAR: Gebruik de jaren uit CONTEXT. Als context leeg is, gebruik 2024.
    
    Output JSON: {{"source": "CSV"|"PDF", "years": [2024], "keywords": ["zoekterm"]}}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system", "content": system_prompt}, {"role":"user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        intent = json.loads(res.choices[0].message.content)
        
        # Update het geheugen van Finny
        if intent.get('years'):
            st.session_state["active_years"] = intent['years']
            
        return intent
    except:
        return {"source": "PDF", "keywords": [], "years": st.session_state["active_years"] or ['2024']}

def query_csv_universal(data, intent):
    """Zoekmachine."""
    if data["trans"] is None: return "Geen transacties.", []
    
    df = data["trans"].copy()
    keywords = intent.get('keywords', [])
    years = intent.get('years', [])
    debug_info = []

    # 1. Jaar Filter
    if years and 'Year_Clean' in df.columns:
        years_str = [str(y) for y in years]
        df = df[df['Year_Clean'].isin(years_str)]
        debug_info.append(f"📅 Jaren: {years_str}")

    if not keywords:
        return "Geen zoektermen.", debug_info

    # 2. Codes Zoeken
    target_codes = set()
    for k in keywords:
        word = k.lower()
        # Synoniemen
        if data["syn"] is not None:
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(word, na=False)]
            if not matches.empty:
                found = matches['Finny_GLCode'].tolist()
                target_codes.update(found)
                debug_info.append(f"✅ Synoniem '{word}' -> {found}")
        # RGS
        if data["rgs"] is not None:
            matches = data["rgs"][data["rgs"]['SearchBlob'].str.contains(word, na=False)]
            if not matches.empty:
                found = matches['Finny_GLCode'].tolist()
                target_codes.update(found)
                debug_info.append(f"✅ RGS '{word}' -> {found}")

    # 3. Filteren
    mask_code = pd.Series([False] * len(df), index=df.index)
    if target_codes:
        mask_code = df['Finny_GLCode'].isin(list(target_codes))

    pattern = '|'.join([re.escape(k.lower()) for k in keywords])
    mask_text = df['UniversalSearch'].str.contains(pattern, na=False)

    df_final = df[mask_code | mask_text]

    # 4. Resultaat (Gebruik Markdown tabel voor nette weergave)
    count = len(df_final)
    total = df_final['AmountDC_num'].sum()
    
    if count == 0:
        return f"Geen data gevonden voor '{keywords}' in {years}.", debug_info
    
    res = f"Gevonden: {count} transacties.\nTotaal: € {total:,.2f}\n\n"
    
    cols = ['EntryDate', 'Description', 'AccountName', 'Finny_GLDescription', 'AmountDC_num']
    valid_cols = [c for c in cols if c in df_final.columns]
    
    limit = 100
    # Hier gebruiken we to_markdown() wat tabulate nodig heeft
    res += df_final[valid_cols].head(limit).to_markdown(index=False, floatfmt=".2f")
            
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
        # LOGO
        image_files = glob.glob("*.png") + glob.glob("*.jpg")
        if image_files:
            st.image(image_files[0], width=150)
        else:
            st.title("Finny") # Fallback tekst
            
        st.markdown("### Status")
        if data["trans"] is not None: 
            st.success(f"Transacties: {len(data['trans'])}")
            # LAAT ZIEN WELKE JAREN ER ZIJN
            if data["years_found"]:
                st.info(f"Beschikbare jaren: {', '.join(data['years_found'])}")
        
        if st.button("Reset"): 
            st.session_state["active_years"] = []
            st.rerun()

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
                    st.caption(f"Bron: PDF | Focus: {intent['years']}")
                else:
                    res_text, debug_log = query_csv_universal(data, intent)
                    context = res_text
                    with st.expander("🔍 Zoekdetails"):
                        st.write(f"**Zoektermen:** {intent['keywords']} | **Jaren:** {intent['years']}")
                        for step in debug_log: st.write(step)
                
                sys_msg = """
                Je bent Finny. 
                Gebruik de CONTEXT data voor je antwoord.
                - De context bevat al een tabel en totalen. Gebruik die.
                - Maak er een vriendelijk, zakelijk antwoord van.
                """
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
