import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader

# Probeer OpenAI, installeer indien nodig (failsafe)
try:
    from openai import OpenAI
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="Finny 11.0 | Strict Logic", page_icon="📏", layout="wide")

# Client setup
client = None
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ==========================================
# 2. DATA LADEN (NORMALISATIE IS KEY)
# ==========================================
@st.cache_data
def load_data():
    data = {"trans": None, "gl": None, "pdf": ""}

    # A. GROOTBOEKSCHEMA (De Bron van Waarheid)
    if os.path.exists("Finny_GL_Lite.csv"):
        try:
            # Lees in
            df_gl = pd.read_csv("Finny_GL_Lite.csv", sep=";", on_bad_lines='skip', dtype=str)
            
            # Kolommen opschonen (spaties weg)
            df_gl.columns = df_gl.columns.str.strip()
            
            # Code normaliseren (4310.0 -> 4310)
            if 'Finny_GLCode' in df_gl.columns:
                df_gl['Finny_GLCode'] = df_gl['Finny_GLCode'].str.split('.').str[0].str.strip()
            
            data["gl"] = df_gl
        except Exception as e:
            st.error(f"Fout in GL CSV: {e}")

    # B. TRANSACTIES (De Details)
    if os.path.exists("Finny_Transactions_Lite.csv"):
        try:
            df_t = pd.read_csv("Finny_Transactions_Lite.csv", sep=";", on_bad_lines='skip', low_memory=False)
            df_t.columns = df_t.columns.str.strip()
            
            # Code normaliseren (MOET matchen met GL)
            if 'Finny_GLCode' in df_t.columns:
                df_t['Finny_GLCode'] = df_t['Finny_GLCode'].astype(str).str.split('.').str[0].str.strip()
                
            # Bedrag numeriek maken (komma/punt fix)
            if 'AmountDC_num' in df_t.columns:
                df_t['AmountDC_num'] = pd.to_numeric(df_t['AmountDC_num'], errors='coerce').fillna(0.0)
                
            # Jaar numeriek
            if 'Finny_Year' in df_t.columns:
                 df_t['Finny_Year'] = pd.to_numeric(df_t['Finny_Year'], errors='coerce').fillna(0).astype(int)
                 
            data["trans"] = df_t
        except Exception as e:
            st.error(f"Fout in Transactie CSV: {e}")

    # C. PDF (Voor Winst/Balans)
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for i in range(min(15, len(reader.pages))):
                text += reader.pages[i].extract_text()
            data["pdf"] += f"\n--- {pdf} ---\n{text}"
        except: pass
            
    return data

# ==========================================
# 3. DE STRICTE LOGICA (ZOALS JE BESCHREEF)
# ==========================================
def execute_logic(question, data):
    log = [] # We houden stap voor stap bij wat we doen
    
    if not client: return "Geen API key", []
    
    # STAP 1: Vertaal vraag naar ZOEKTERM (voor de GL lijst)
    # We vragen AI niet om de code, maar om het WOORD waarop we moeten zoeken in Kolom B.
    prompt_term = f"""
    Vertaal de vraag naar een zoekterm voor het grootboekschema.
    Vraag: "{question}"
    
    Voorbeeld: "Wat zijn de telefoonkosten?" -> "Telefoon"
    Voorbeeld: "Hoeveel kosten aan auto?" -> "Auto"
    Voorbeeld: "Omzet 2023?" -> "Omzet"
    
    Geef alleen het woord.
    """
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt_term}])
    search_term = res.choices[0].message.content.strip().replace('"', '')
    log.append(f"1. Zoekterm bepaald: '{search_term}'")
    
    # STAP 2: Zoek in GL Tabel (Kolom B -> Vind Kolom A)
    found_codes = []
    found_names = []
    
    if data["gl"] is not None:
        # Zoek in Description (case insensitive)
        mask = data["gl"]['Finny_GLDescription'].str.contains(search_term, case=False, na=False)
        matches = data["gl"][mask]
        
        if not matches.empty:
            found_codes = matches['Finny_GLCode'].unique().tolist()
            found_names = matches['Finny_GLDescription'].unique().tolist()
            log.append(f"2. Gevonden in GL Schema: {len(matches)} rekeningen.")
            log.append(f"   -> Codes: {found_codes}")
            log.append(f"   -> Namen: {found_names}")
        else:
            log.append(f"2. GEEN match in GL schema voor '{search_term}'.")
            # Fallback: Als 'Auto' niet werkt, probeer 'Vervoer'? (Nu even strikt houden)

    # STAP 3: Zoek in Transacties (Met de gevonden codes)
    total_amount = 0.0
    trans_count = 0
    relevant_subset = pd.DataFrame()
    
    if data["trans"] is not None and found_codes:
        # Filter op codes
        subset = data["trans"][data["trans"]['Finny_GLCode'].isin(found_codes)]
        
        # Filter op Jaar (als dat in de vraag zit)
        # Simpele check: zit "2023" in de vraag?
        year_filter = None
        for y in [2022, 2023, 2024, 2025]:
            if str(y) in question:
                year_filter = y
                subset = subset[subset['Finny_Year'] == y]
                log.append(f"3. Filter op Jaar: {y}")
                break
        
        if len(subset) > 0:
            total_amount = subset['AmountDC_num'].sum()
            trans_count = len(subset)
            relevant_subset = subset
            log.append(f"4. Transacties gevonden: {trans_count}")
            log.append(f"   -> Totaalbedrag: € {total_amount:,.2f}")
        else:
            log.append("4. Wel GL codes gevonden, maar geen boekingen in transacties.")
    
    # STAP 4: Formuleer antwoord
    # Als we transacties hebben, is dat het antwoord. Anders kijken we naar PDF.
    
    final_context = ""
    if trans_count > 0:
        final_context = f"""
        UITKOMST UIT CSV:
        Onderwerp: {search_term}
        Rekeningen: {found_names} (Codes: {found_codes})
        Jaar: {year_filter if year_filter else "Alle jaren"}
        TOTAAL BEDRAG: € {total_amount:,.2f}
        
        Details (Top 5):
        {relevant_subset[['EntryDate', 'Description', 'AmountDC_num']].head(5).to_string(index=False)}
        """
    else:
        final_context = "Geen resultaat uit CSV. Kijk in de PDF."

    # Laatste AI slag voor nette zin
    final_prompt = f"""
    Vraag: {question}
    
    Data uit Transacties (CSV):
    {final_context}
    
    Data uit Jaarrekening (PDF):
    {data['pdf'][:5000]}
    
    Geef antwoord. Als CSV data (bedragen) beschikbaar zijn, gebruik die. 
    Als CSV leeg is, gebruik PDF.
    """
    
    res_final = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": final_prompt}])
    return res_final.choices[0].message.content, log

# ==========================================
# 4. UI
# ==========================================
with st.sidebar:
    if os.path.exists("finny_logo.jpg"): st.image("finny_logo.jpg", width=100)
    st.markdown("### ⚙️ Onder de motorkap")
    if st.button("Reset"): st.cache_data.clear(); st.rerun()

st.title("Finny 11.0")
st.markdown("*Logica: Vraag -> Zoek in GL (Kolom B) -> Pak Code (Kolom A) -> Sommeer Transacties.*")

if "auth" not in st.session_state:
    if st.text_input("Wachtwoord", type="password") == "demo2025":
        st.session_state.auth = True
        st.rerun()
else:
    data = load_data()
    if data["gl"] is not None: st.success(f"GL Schema geladen ({len(data['gl'])} rekeningen)")
    if data["trans"] is not None: st.success(f"Transacties geladen ({len(data['trans'])} rijen)")

    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if "log" in m:
                with st.expander("🔍 Zie de gevolgde stappen"):
                    for l in m["log"]: st.code(l, language="text")

    if prompt := st.chat_input("Wat zijn de kosten voor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Logica uitvoeren..."):
                ans, log = execute_logic(prompt, data)
                st.write(ans)
                with st.expander("🔍 Zie de gevolgde stappen", expanded=True):
                    for l in log: st.code(l, language="text")
                st.session_state.messages.append({"role": "assistant", "content": ans, "log": log})
