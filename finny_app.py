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

# 2. DATA LADEN (SCHOON & STRICT)
@st.cache_data
def load_data():
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": ""}
    
    def clean_code(val):
        # Verwijder .0 en spaties (bijv "2600.0" -> "2600")
        return str(val).split('.')[0].strip()

    # A. TRANSACTIES
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", dtype=str)
            # Bedragen naar getal
            if 'AmountDC_num' in df.columns:
                df['AmountDC_num'] = df['AmountDC_num'].str.replace(',', '.').replace('nan', '0')
                df['AmountDC_num'] = pd.to_numeric(df['AmountDC_num'], errors='coerce').fillna(0.0)
            # Codes cleanen voor exacte match
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            # Jaren cleanen
            if 'Finny_Year' in df.columns:
                df['Year_Clean'] = df['Finny_Year'].astype(str).str.split('.').str[0]
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
            # RGS Omschrijving lowercase maken voor matching
            if 'RGS_Omschrijving' in df.columns:
                df['RGS_Omschrijving_Lower'] = df['RGS_Omschrijving'].astype(str).str.lower()
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

# 3. LOGICA: DE IJZEREN KETEN

def get_intent(client, question):
    """
    Stap 0: Haal ALLEEN de kernwoorden en het jaar uit de vraag.
    Geen interpretatie, alleen extractie.
    """
    system_prompt = """
    Je bent een extractor. 
    Haal de volgende elementen uit de vraag:
    1. 'source': 'PDF' (als het gaat om winst/balans/omzet/totaalplaatje) of 'CSV' (kosten/bedragen/transacties).
    2. 'keywords': De specifieke zelfstandige naamwoorden (bijv: "kantoorkosten", "auto", "telefoon").
    3. 'years': De genoemde jaren (bijv: 2024).
    
    Output JSON.
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

def execute_iron_logic(data, intent):
    """
    DE IJZEREN LOGICA (5 STAPPEN)
    """
    if data["trans"] is None: return "Geen transacties geladen.", []
    
    keywords = intent.get('keywords', [])
    years = intent.get('years', [])
    
    log_steps = [] # Hierin houden we het bewijs bij
    
    # STAP 1: Kernwoorden identificeren
    log_steps.append(f"**Stap 1 (Input):** Zoeken naar kernwoorden: `{keywords}` in jaren `{years}`")
    
    if not keywords:
        return "Ik kan geen kernwoorden uit je vraag halen om op te zoeken.", log_steps

    found_codes = set()

    # Loop door elk zoekwoord
    for word in keywords:
        word = word.lower()
        match_found_for_word = False
        
        # STAP 2: Zoek in Synoniemenlijst
        if data["syn"] is not None:
            # Zoek exacte of deel-match in de kolom Synoniem
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(word, na=False)]
            
            if not matches.empty:
                # STAP 3: Pak de bijbehorende Code (Finny_GLCode)
                codes = matches['Finny_GLCode'].unique().tolist()
                # Pak ook de omschrijving erbij voor de log
                desc = matches['Finny_GLDescription'].unique().tolist()
                
                found_codes.update(codes)
                log_steps.append(f"**Stap 2 & 3 (Synoniem):** Woord '{word}' gevonden in synoniemenlijst. -> Koppeling met categorie '{desc}' -> Code(s): `{codes}`")
                match_found_for_word = True
        
        # Als niet gevonden in synoniemen, probeer RGS Omschrijving (Fallback binnen stap 2/3)
        if not match_found_for_word and data["rgs"] is not None:
            matches = data["rgs"][data["rgs"]['RGS_Omschrijving_Lower'].str.contains(word, na=False)]
            if not matches.empty:
                codes = matches['Finny_GLCode'].unique().tolist()
                found_codes.update(codes)
                log_steps.append(f"**Stap 2 & 3 (RGS):** Woord '{word}' gevonden in RGS Omschrijving. -> Code(s): `{codes}`")
                match_found_for_word = True
        
        if not match_found_for_word:
            log_steps.append(f"❌ **Stap 2 & 3:** Woord '{word}' komt niet voor in Synoniemenlijst én niet in RGS. Ik stop met zoeken voor dit woord.")

    if not found_codes:
        return "Geen enkele GL-code gevonden op basis van je zoekwoorden. Ik kan stap 4 (filteren) niet uitvoeren.", log_steps

    # STAP 4: Filter Transacties op Code
    df = data["trans"].copy()
    
    # Filter op Jaar (indien van toepassing)
    if years and 'Year_Clean' in df.columns:
        years_str = [str(y) for y in years]
        df = df[df['Year_Clean'].isin(years_str)]
    
    # Filter op de gevonden Codes (DE HARDE FILTER)
    mask = df['Finny_GLCode'].isin(list(found_codes))
    df_final = df[mask]
    
    count = len(df_final)
    log_steps.append(f"**Stap 4 (Transacties):** Filteren op codes `{list(found_codes)}` in jaren `{years}`. -> Resultaat: {count} transacties gevonden.")

    # STAP 5: Rekenen
    if count == 0:
        return "Wel codes gevonden, maar geen boekingen in dit jaar op deze codes.", log_steps
        
    total = df_final['AmountDC_num'].sum()
    log_steps.append(f"**Stap 5 (Rekenen):** Subtotaal berekend: € {total:,.2f}")

    # Resultaat opmaken
    res = f"**Totaal:** € {total:,.2f}\n\n"
    
    cols = ['EntryDate', 'Description', 'Finny_GLCode', 'AmountDC_num']
    valid_cols = [c for c in cols if c in df_final.columns]
    
    if count < 100:
        res += df_final[valid_cols].to_string(index=False)
    else:
        res += f"(Lijst te lang: {count} regels. Hier is de top 10)\n"
        res += df_final[valid_cols].head(10).to_string(index=False)
            
    return res, log_steps

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
        if os.path.exists("finny_logo.png"):
            st.image("finny_logo.png", width=150)
        st.title("🦁 Finny")
        if data["trans"] is not None: st.success(f"Transacties: {len(data['trans'])}")
        if data["syn"] is not None: st.success(f"Synoniemen: {len(data['syn'])}")
        if data["rgs"] is not None: st.success(f"RGS: {len(data['rgs'])}")
        if st.button("Reset"): st.rerun()

    st.title("Finny Demo - IJzeren Logica")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Vraag Finny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Stappenplan uitvoeren..."):
                # 1. Router
                intent = get_intent(client, prompt)
                
                context = ""
                if intent['source'] == "PDF":
                    context = data["pdf_text"]
                    st.caption("Bron: PDF")
                    # PDF Logic (Simpel)
                    sys_msg = "Je bent Finny. Geef antwoord op basis van de PDF context."
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

                else:
                    # CSV Logic (De IJzeren Keten)
                    res_text, log = execute_iron_logic(data, intent)
                    
                    # Toon EERST de logica (zoals gevraagd)
                    st.markdown("### ⚙️ Finny's Logica")
                    for step in log:
                        st.write(step)
                    st.markdown("---")
                    
                    # Toon dan het resultaat
                    st.markdown("### 📊 Resultaat")
                    st.text(res_text)
                    
                    # Opslaan voor chat history
                    final_msg = "Zie hierboven de berekening en details."
                    st.session_state.messages.append({"role": "assistant", "content": final_msg})
