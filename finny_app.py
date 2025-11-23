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

# Initialiseer sessie state voor context-geheugen
if "active_years" not in st.session_state:
    st.session_state["active_years"] = []
if "active_category" not in st.session_state:
    st.session_state["active_category"] = None

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
        # Verwijder .0 en spaties
        return str(val).split('.')[0].strip()

    # A. TRANSACTIES
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", dtype=str)
            # Bedragen naar getal
            if 'AmountDC_num' in df.columns:
                df['AmountDC_num'] = df['AmountDC_num'].str.replace(',', '.').replace('nan', '0')
                df['AmountDC_num'] = pd.to_numeric(df['AmountDC_num'], errors='coerce').fillna(0.0)
            # Codes & Jaren cleanen
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            if 'Finny_Year' in df.columns:
                df['Year_Clean'] = df['Finny_Year'].astype(str).str.split('.').str[0]

            # Universal Search kolom maken
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

    # C. RGS (Met Referentiecode voor hiërarchie)
    if os.path.exists("Finny_RGS.csv"):
        try:
            df = pd.read_csv("Finny_RGS.csv", sep=";", dtype=str)
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            
            # SearchBlob maken
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

# 3. LOGICA: INTENT & ROUTER (MET GEHEUGEN)

def get_intent(client, question):
    """
    Bepaalt Bron, Jaar, Keywords én Categorie.
    Gebruikt sessie-geheugen als fallback.
    """
    
    # Huidige context ophalen voor de prompt
    prev_years = st.session_state["active_years"]
    prev_cat = st.session_state["active_category"]
    
    system_prompt = f"""
    Je bent de router van Finny. Analyseer de vraag.
    
    Huidige Context (uit vorig bericht):
    - Jaren: {prev_years}
    - Categorie: {prev_cat}
    
    TAAK:
    1. 'source': 'PDF' (winst/balans/omzet/totaal) of 'CSV' (kosten/details/leveranciers).
    2. 'keywords': Specifieke zoekwoorden uit de vraag.
    3. 'years': Jaren in de vraag. ALS GEEN JAAR GENOEMD: Gebruik de jaren uit de context.
    4. 'category': De algemene boekhoudkundige categorie (bijv. "Huisvesting", "Communicatie", "Autokosten").
    
    Output JSON: {{"source": "CSV"|"PDF", "years": [2024], "keywords": ["..."], "category": "..."}}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system", "content": system_prompt}, {"role":"user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        intent = json.loads(res.choices[0].message.content)
        
        # Update Sessie Geheugen
        if intent.get('years'):
            st.session_state["active_years"] = intent['years']
        
        if intent.get('category'):
            st.session_state["active_category"] = intent['category']
            
        return intent
    except:
        return {"source": "PDF", "keywords": [], "years": [], "category": "Algemeen"}

def query_csv_advanced(data, intent):
    """
    De Verbeterde Zoekmachine (CSV).
    - Gebruikt RGS Hiërarchie (Parent/Child).
    - Gebruikt Categorie als extra zoekterm.
    - Onderscheidt Code-match vs Tekst-match.
    """
    if data["trans"] is None: return "Geen transacties geladen.", []
    
    # 1. Verzamel zoektermen (Keywords + Categorie)
    keywords = intent.get('keywords', [])
    category = intent.get('category')
    if category and category.lower() not in keywords:
        keywords.append(category)
        
    years = intent.get('years', [])
    # Fallback op sessie als leeg (zou door router al gedaan moeten zijn, maar dubbelcheck)
    if not years: years = st.session_state["active_years"]

    log_steps = []
    log_steps.append(f"**Stap 1 (Input):** Zoeken naar `{keywords}` in jaren `{years}`")

    if not keywords:
        return "Ik kan geen specifiek onderwerp vinden in je vraag.", log_steps

    # 2. CODES ZOEKEN (Met RGS Hiërarchie)
    found_codes = set()
    
    for word in keywords:
        word = word.lower()
        
        # A. Synoniemen
        if data["syn"] is not None:
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(word, na=False)]
            if not matches.empty:
                codes = matches['Finny_GLCode'].unique().tolist()
                found_codes.update(codes)
                log_steps.append(f"✅ **Synoniem:** '{word}' -> Code(s): `{codes}`")
        
        # B. RGS (Met Parent/Child Logic)
        if data["rgs"] is not None:
            # Zoek eerst regels die matchen op tekst
            matches = data["rgs"][data["rgs"]['SearchBlob'].str.contains(word, na=False)]
            
            if not matches.empty:
                # Voor elke match, kijk naar de Referentiecode (Hiërarchie)
                for idx, row in matches.iterrows():
                    direct_code = row['Finny_GLCode']
                    ref_code = str(row.get('RGS_Referentiecode', ''))
                    
                    # Voeg directe code toe
                    found_codes.add(direct_code)
                    
                    # RGS Hiërarchie uitbreiding
                    # Als code bijv W.B.1400 is, zoeken we alles dat begint met W.B.1400 of W.B (afhankelijk van diepte)
                    if len(ref_code) > 4 and '.' in ref_code:
                        # Pak de parent (strip het laatste deel na de punt)
                        # Bijv: W.W.Sup.Kui.Kan -> W.W.Sup.Kui
                        parent_ref = ref_code.rsplit('.', 1)[0]
                        
                        # Zoek alle GL codes die onder deze parent vallen
                        child_rows = data["rgs"][data["rgs"]['RGS_Referentiecode'].astype(str).str.startswith(parent_ref)]
                        child_codes = child_rows['Finny_GLCode'].unique().tolist()
                        
                        if len(child_codes) > 1:
                            found_codes.update(child_codes)
                            log_steps.append(f"🔹 **RGS Tak:** '{word}' matcht '{ref_code}'. Uitgebreid naar hele groep '{parent_ref}' -> {len(child_codes)} codes.")
                    
    # 3. FILTEREN TRANSACTIES
    df = data["trans"].copy()
    
    # Jaar Filter
    if years and 'Year_Clean' in df.columns:
        years_str = [str(y) for y in years]
        df = df[df['Year_Clean'].isin(years_str)]
        log_steps.append(f"📅 Gefilterd op jaren: {years}")

    # Splitsing: Code Match vs Tekst Match (voor debugging en precisie)
    mask_code = pd.Series([False] * len(df), index=df.index)
    mask_text = pd.Series([False] * len(df), index=df.index)

    # A. Code Match
    if found_codes:
        mask_code = df['Finny_GLCode'].isin(list(found_codes))
    
    # B. Tekst Match (Vangnet)
    pattern = '|'.join([re.escape(k.lower()) for k in keywords])
    mask_text = df['UniversalSearch'].str.contains(pattern, na=False)

    # Combineer
    df_final = df[mask_code | mask_text].copy()
    
    # Tag de bron van de match (voor debug)
    df_final.loc[mask_code, 'Match_Type'] = 'Code'
    df_final.loc[(~mask_code) & mask_text, 'Match_Type'] = 'Tekst'

    count = len(df_final)
    total = df_final['AmountDC_num'].sum()
    
    # Analyseer resultaten voor debug
    code_hits = df_final[df_final['Match_Type'] == 'Code'].shape[0]
    text_hits = df_final[df_final['Match_Type'] == 'Tekst'].shape[0]
    
    log_steps.append(f"📊 **Resultaat:** {count} transacties. ({code_hits} via GL-Code, {text_hits} via Tekstmatch)")
    
    if text_hits > 0:
        # Laat zien welke beschrijvingen alleen via tekst zijn gevonden (potentieel nieuwe synoniemen)
        sample_text = df_final[df_final['Match_Type'] == 'Tekst']['Description'].unique()[:3]
        log_steps.append(f"💡 **Tip:** Deze omschrijvingen werden gevonden op tekst (mist GL koppeling): `{list(sample_text)}`")

    if count == 0:
        return f"Geen data gevonden voor '{keywords}' in {years}.", log_steps

    # Output maken
    res = f"Gevonden: {count} transacties.\nTotaal: € {total:,.2f}\n"
    
    cols = ['EntryDate', 'Description', 'AccountName', 'Finny_GLDescription', 'AmountDC_num']
    valid_cols = [c for c in cols if c in df_final.columns]
    
    if count < 100:
        res += f"\nLijst:\n{df_final[valid_cols].to_string(index=False)}"
    else:
        res += f"(Lijst lang, top 20 getoond)\n{df_final[valid_cols].head(20).to_string(index=False)}"
            
    return res, log_steps

# 4. DE APP UI
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
        # Logo (Flexibel)
        image_files = glob.glob("finny_logo.*") # Pakt png of jpg
        if image_files:
            st.image(image_files[0], width=150)
        st.title("🦁 Finny")
        
        # Status indicatoren
        if data["trans"] is not None: st.success(f"Transacties: {len(data['trans'])}")
        
        # Context Debugger in Sidebar
        st.markdown("---")
        st.markdown("**Actief Geheugen:**")
        st.markdown(f"📅 Jaar: `{st.session_state['active_years']}`")
        st.markdown(f"📂 Categorie: `{st.session_state['active_category']}`")
        
        if st.button("Reset"): 
            st.session_state["active_years"] = []
            st.session_state["active_category"] = None
            st.rerun()

    st.title("Finny Demo")
    
    # Chat Historie Tonen
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Vraag Finny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("..."):
                # 1. Intent & Router
                intent = get_intent(client, prompt)
                
                context = ""
                # 2. Data Ophalen
                if intent['source'] == "PDF":
                    context = data["pdf_text"]
                    st.caption("Bron: PDF (Jaarrekening)")
                else:
                    res_text, debug_log = query_csv_advanced(data, intent)
                    context = res_text
                    
                    # Debug Expander
                    with st.expander("🔍 Finny's Zoektocht"):
                        st.write(f"**Intent:** {intent}")
                        for step in debug_log: st.write(step)
                
                # 3. LLM Antwoord (Met chatgeschiedenis)
                # We bouwen de message history op voor de LLM
                messages_for_llm = [
                    {"role": "system", "content": f"Je bent Finny. Gebruik deze CONTEXT data voor je antwoord:\n\n{context}\n\nAls er staat 'Totaal: ...', gebruik dat. Reken niet zelf."}
                ]
                
                # Voeg laatste 6 berichten toe voor conversational memory
                for msg in st.session_state.messages[-6:]:
                    messages_for_llm.append({"role": msg["role"], "content": msg["content"]})
                
                # Voeg huidige vraag toe (als die nog niet in history zat, maar dat zit ie wel door st.session_state append hierboven)
                # Omdat we history pakken waar de user prompt net is toegevoegd, is dit goed.
                
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_for_llm
                )
                
                reply = res.choices[0].message.content
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
