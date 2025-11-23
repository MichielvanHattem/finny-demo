import streamlit as st
import pandas as pd
import os
import json
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# 1. CONFIGURATIE
st.set_page_config(page_title="Finny", page_icon="🦁", layout="wide")
load_dotenv()

# Wachtwoord
def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="pw", on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# 2. DATA LADEN (PUUR LEZEN, GEEN POETSWERK)
@st.cache_data
def load_data():
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": ""}
    
    # A. TRANSACTIES (Vertrouwen op schone data)
    if os.path.exists("Finny_Transactions.csv"):
        try:
            # We lezen met puntkomma. We gaan ervan uit dat AmountDC_num een getal is.
            # decimal=',' zorgt dat 12,50 wordt gelezen als 12.50
            df = pd.read_csv("Finny_Transactions.csv", sep=";", decimal=",")
            data["trans"] = df
        except Exception as e:
            st.error(f"Fout transacties: {e}")

    # B. SYNONIEMEN
    if os.path.exists("Finny_Synonyms.csv"):
        try:
            df_syn = pd.read_csv("Finny_Synonyms.csv", sep=";")
            # Zorg dat synoniemen lowercase zijn voor matching
            df_syn['Synoniem'] = df_syn['Synoniem'].astype(str).str.lower()
            data["syn"] = df_syn
        except: pass

    # C. RGS (Optioneel voor context)
    if os.path.exists("Finny_RGS.csv"):
        try:
            data["rgs"] = pd.read_csv("Finny_RGS.csv", sep=";")
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

# 3. LOGICA (ROUTER & ENGINE)

def get_intent(client, question):
    """Bepaalt bron: PDF of CSV."""
    system_prompt = """
    Je bent de router. Bepaal de bron.
    1. 'PDF' -> Jaarrekening, balans, winst, strategie.
    2. 'CSV' -> Kosten, bedragen, leveranciers, details.
    
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

def query_clean_data(data, intent):
    """
    Zoekt direct in de schone data.
    Logica: Zoekterm -> Synoniemenlijst -> GL Code -> Transacties
    """
    if data["trans"] is None: return "Geen transacties."
    
    df = data["trans"].copy()
    keywords = intent.get('keywords', [])
    years = intent.get('years', [])
    
    # 1. Filter op Jaar (Finny_Year kolom in clean data)
    if years and 'Finny_Year' in df.columns:
        df = df[df['Finny_Year'].isin(years)]

    # 2. Filter op Inhoud
    if keywords:
        gl_codes = []
        # A. Kijk in Synoniemen
        if data["syn"] is not None:
            for k in keywords:
                # Exacte of deel-match in synoniemen
                matches = data["syn"][data["syn"]['Synoniem'].str.contains(k.lower(), na=False)]
                if not matches.empty:
                    gl_codes.extend(matches['Finny_GLCode'].tolist())
        
        if gl_codes:
            # B. We hebben codes gevonden -> Filter transacties op code
            # Zorg dat types matchen (string vs int), voor zekerheid naar string converten
            df['Finny_GLCode'] = df['Finny_GLCode'].astype(str).str.replace('.0', '')
            gl_codes_str = [str(c).replace('.0', '') for c in gl_codes]
            
            df = df[df['Finny_GLCode'].isin(gl_codes_str)]
        else:
            # C. Geen synoniem? Dan plat zoeken in Omschrijving (Description)
            pattern = '|'.join(keywords)
            df = df[df['Description'].astype(str).str.contains(pattern, case=False, na=False)]

    # 3. Resultaat
    # We gebruiken AmountDC_num omdat dat de schone numerieke kolom is
    if 'AmountDC_num' in df.columns:
        total = df['AmountDC_num'].sum()
    else:
        total = 0.0 # Fallback als kolomnaam toch anders is
        
    count = len(df)
    
    if count == 0: return f"Geen data gevonden voor {keywords}."
    
    res = f"Aantal transacties: {count}\nTotaal: € {total:,.2f}\n"
    
    # Toon details (Description en Amount)
    cols_to_show = [c for c in ['EntryDate', 'Description', 'AmountDC_num'] if c in df.columns]
    
    if count < 20:
        res += f"\nDetails:\n{df[cols_to_show].to_string(index=False)}"
    else:
        # Top 5
        if 'Description' in df.columns:
            top = df.groupby('Description')['AmountDC_num'].sum().sort_values().head(5)
            res += f"\nTop 5 Posten:\n{top.to_string()}"
            
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
                intent = get_intent(client, prompt)
                st.caption(f"Strategie: {intent['source']} | Zoektermen: {intent['keywords']}")
                
                context = ""
                if intent['source'] == "PDF":
                    context = data["pdf_text"]
                else:
                    context = query_clean_data(data, intent)
                
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
