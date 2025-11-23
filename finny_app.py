import streamlit as st
import pandas as pd
import os
import json
import re
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# 1. CONFIGURATIE
st.set_page_config(page_title="Finny", page_icon="🦁", layout="wide")
load_dotenv()

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="pw", on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# 2. DATA LADEN (SCHOON & SIMPEL)
@st.cache_data
def load_data():
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": ""}
    
    # A. CSV BESTANDEN (Jouw 3 schone bronnen)
    try:
        # Transacties
        if os.path.exists("Finny_Transactions.csv"):
            df = pd.read_csv("Finny_Transactions.csv", sep=";", dtype={'Finny_GLCode': str})
            # Zorg dat bedrag een getal is (punt als decimaal)
            if df['AmountDC_num'].dtype == object:
                df['AmountDC_num'] = pd.to_numeric(df['AmountDC_num'].astype(str).str.replace(',', '.'), errors='coerce')
            data["trans"] = df.fillna("") # Lege waarden opvullen voorkomt crashes

        # Synoniemen
        if os.path.exists("Finny_Synonyms.csv"):
            df_syn = pd.read_csv("Finny_Synonyms.csv", sep=";", dtype=str)
            df_syn['Synoniem'] = df_syn['Synoniem'].str.lower().str.strip()
            data["syn"] = df_syn

        # RGS
        if os.path.exists("Finny_RGS.csv"):
            df_rgs = pd.read_csv("Finny_RGS.csv", sep=";", dtype=str)
            data["rgs"] = df_rgs

    except Exception as e:
        st.error(f"Data Fout: {e}")

    # B. PDF BESTANDEN
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages[:15]: text += page.extract_text()
            data["pdf_text"] += f"\n--- BRON: {pdf} ---\n{text[:8000]}"
        except: pass
        
    return data

# 3. LOGICA (DE CORRECTE ZOEKSTRATEGIE)

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
    De En/En Strategie:
    1. Zoek GL-codes via Synoniemen.
    2. Zoek GL-codes via RGS omschrijvingen.
    3. Filter transacties op: (GL-code match) OF (Tekst match in omschrijving).
    """
    if data["trans"] is None: return "Geen transacties."
    
    df = data["trans"].copy()
    keywords = intent.get('keywords', [])
    years = intent.get('years', [])
    
    # 1. Filter op Jaar
    if years and 'Finny_Year' in df.columns:
        # Zorg dat jaar kolom numeric is voor matching
        df['Finny_Year'] = pd.to_numeric(df['Finny_Year'], errors='coerce')
        df = df[df['Finny_Year'].isin(years)]

    if not keywords:
        return "Geen zoektermen opgegeven."

    # 2. Verzamel GL Codes (Via Synoniemen & RGS)
    target_codes = set()
    
    for k in keywords:
        word = k.lower()
        
        # Check Synoniemen
        if data["syn"] is not None:
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(word, na=False)]
            target_codes.update(matches['Finny_GLCode'].tolist())
            
        # Check RGS (Omschrijvingen)
        if data["rgs"] is not None:
            matches = data["rgs"][data["rgs"]['RGS_Omschrijving'].str.lower().str.contains(word, na=False)]
            target_codes.update(matches['Finny_GLCode'].tolist())

    # 3. De Grote Filter (Code OF Tekst)
    # We maken een masker (True/False lijst) voor de rijen die we willen
    
    # A. Match op GL Code (als we die gevonden hebben)
    mask_code = pd.Series([False] * len(df), index=df.index) # Start met alles False
    if target_codes:
        # Zorg dat types matchen (string vs string)
        clean_targets = [str(c).split('.')[0] for c in target_codes] # '2600.0' -> '2600'
        df['Finny_GLCode_clean'] = df['Finny_GLCode'].astype(str).str.split('.').str[0]
        mask_code = df['Finny_GLCode_clean'].isin(clean_targets)

    # B. Match op Tekst (Description)
    pattern = '|'.join([re.escape(k) for k in keywords])
    mask_text = df['Description'].astype(str).str.contains(pattern, case=False, na=False)

    # C. Combineer: Code Match OF Tekst Match
    df_final = df[mask_code | mask_text]

    # 4. Resultaat
    count = len(df_final)
    if count == 0: return f"Geen data gevonden voor '{keywords}' in {years}."
    
    total = df_final['AmountDC_num'].sum()
    
    res = f"Zoekterm: {keywords}\nGevonden codes: {list(target_codes)[:5]}\n"
    res += f"Aantal transacties: {count}\nTotaal: € {total:,.2f}\n"
    
    # Toon relevante kolommen
    cols = ['EntryDate', 'Description', 'AmountDC_num', 'Finny_GLDescription']
    
    if count < 30:
        res += f"\nDetails:\n{df_final[cols].to_string(index=False)}"
    else:
        # Top 5 als het veel is
        top = df_final.groupby('Description')['AmountDC_num'].sum().sort_values().head(5)
        res += f"\nTop 5 Kostenposten:\n{top.to_string()}"
            
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
        # Logo check
        for ext in ["jpg", "png", "jpeg"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.title("🦁 Finny")
        if data["trans"] is not None: st.success(f"Transacties: {len(data['trans'])}")
        if data["syn"] is not None: st.success(f"Synoniemen: {len(data['syn'])}")
        if st.button("Reset"): st.rerun()

    st.title("Finny Demo")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Vraag Finny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("..."):
                # 1. Router
                intent = get_intent(client, prompt)
                
                # 2. Data
                context = ""
                if intent['source'] == "PDF":
                    st.caption("Strategie: PDF (Jaarrekening)")
                    context = data["pdf_text"]
                else:
                    st.caption(f"Strategie: CSV (Transacties) | Zoekt: {intent['keywords']}")
                    context = query_csv_exact(data, intent)
                
                # 3. Antwoord
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
