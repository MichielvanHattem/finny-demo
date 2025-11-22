import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
from datetime import datetime

# ==========================================
# 1. INSTELLINGEN & TOOLS
# ==========================================
st.set_page_config(page_title="Finny 6.0 | Transparant", page_icon="🦁", layout="wide")

def normalize_gl_code(val):
    """Maakt van alles (4350.0, '4350', 4350) een schone string '4350'"""
    try:
        return str(float(val)).split('.')[0].strip()
    except:
        return str(val).strip()

# ==========================================
# 2. DATA LADEN (MET LOGO HERSTEL)
# ==========================================
@st.cache_data
def load_data():
    data = {"trans": None, "syllabus": "", "pdf_text": "", "years": []}
    
    # 1. Transacties
    if os.path.exists("Finny_Transactions_Lite.csv"):
        try:
            df = pd.read_csv("Finny_Transactions_Lite.csv", sep=";", on_bad_lines='skip', low_memory=False)
            
            # --- KRITIEKE STAP: NORMALISATIE ---
            # We maken een speciale 'MATCH_CODE' kolom die ALTIJD schoon is (geen .0, geen spaties)
            if 'Finny_GLCode' in df.columns:
                df['MATCH_CODE'] = df['Finny_GLCode'].apply(normalize_gl_code)
            
            # Datums en Jaren
            if 'EntryDate' in df.columns:
                df['EntryDate'] = pd.to_datetime(df['EntryDate'], errors='coerce')
            if 'Finny_Year' in df.columns:
                df['Finny_Year'] = pd.to_numeric(df['Finny_Year'], errors='coerce').fillna(0).astype(int)
                data["years"] = sorted(df[df['Finny_Year'] > 2000]['Finny_Year'].unique().tolist())
                
            data["trans"] = df
        except Exception as e:
            st.error(f"Fout CSV: {e}")

    # 2. Syllabus
    if os.path.exists("Finny_syllabus_v9_4_with_B3_B4_B5.txt"):
        with open("Finny_syllabus_v9_4_with_B3_B4_B5.txt", "r", encoding="utf-8") as f:
            data["syllabus"] = f.read()
            
    # 3. PDF
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for i, page in enumerate(reader.pages):
                if i < 10: text += page.extract_text()
            data["pdf_text"] += f"--- {pdf} ---\n{text[:4000]}\n"
        except: pass
        
    return data

# ==========================================
# 3. SLIMME LOGICA (MET RAPPORTAGE)
# ==========================================
def get_search_plan(client, question, syllabus, years):
    """Vraagt GPT om een zoekplan"""
    years_str = ", ".join(map(str, years))
    prompt = f"""
    Je bent Finny. Analyseer de vraag.
    CONTEXT:
    - Beschikbare Jaren: {years_str} (Kies de meest recente als geen jaar genoemd).
    - Syllabus: {syllabus[:2000]}
    
    VRAAG: "{question}"
    
    TAAK:
    Welke GL Codes (rekeningnummers) moet ik zoeken in de CSV?
    
    ANTWOORD JSON:
    {{
        "source": "CSV" of "PDF",
        "gl_codes": [lijst van nummers, bijv 4350, 4360],
        "search_terms": ["tekst", "tekst"],
        "years": [2023, 2024],
        "reason": "korte uitleg"
    }}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"source": "BOTH", "gl_codes": [], "years": []}

def execute_search_with_report(plan, df):
    """Voert zoekopdracht uit én houdt een logboek bij"""
    report = [] # Hier slaan we op wat er gebeurt
    
    if df is None:
        return "Geen data", ["❌ DataFrame is leeg."]
    
    subset = df.copy()
    report.append(f"1️⃣ Start: {len(subset)} regels in totaal.")
    
    # STAP A: Filter Jaar
    req_years = plan.get('years', [])
    if req_years:
        subset = subset[subset['Finny_Year'].isin(req_years)]
        report.append(f"2️⃣ Na filter op jaren {req_years}: {len(subset)} regels over.")
    else:
        report.append("2️⃣ Geen jaren gespecificeerd, alle jaren behouden.")
        
    # STAP B: Filter Codes (De cruciale stap)
    req_codes = plan.get('gl_codes', [])
    if req_codes:
        # Zet de gevraagde codes ook om naar schone strings
        clean_reqs = [normalize_gl_code(c) for c in req_codes]
        report.append(f"3️⃣ Zoekopdracht vertaald naar codes: {clean_reqs}")
        
        # Match op onze schone MATCH_CODE kolom
        subset = subset[subset['MATCH_CODE'].isin(clean_reqs)]
        
        if len(subset) == 0:
            # PROBEER IETS ANDERS: Wildcard match?
            report.append("   ⚠️ Directe match leverde 0 op. Ik check of de code ergens in de string zit...")
            pattern = '|'.join(clean_reqs)
            subset = df[df['MATCH_CODE'].str.contains(pattern, na=False)]
            
        report.append(f"   -> Resultaat na code filter: {len(subset)} regels.")
    else:
        report.append("3️⃣ Geen GL codes in plan. Ik zoek alleen op tekst.")

    # STAP C: Filter Tekst (optioneel)
    req_terms = plan.get('search_terms', [])
    if req_terms and len(subset) > 0:
        pat = '|'.join(req_terms)
        # We zoeken breed
        subset = subset[
            subset['Description'].astype(str).str.contains(pat, case=False) |
            subset['AccountName'].astype(str).str.contains(pat, case=False) |
            subset['Finny_GLDescription'].astype(str).str.contains(pat, case=False)
        ]
        report.append(f"4️⃣ Na tekstfilter '{req_terms}': {len(subset)} regels.")

    # RESULTAAT
    total = subset['AmountDC_num'].sum()
    
    if len(subset) > 0:
        # Maak een mooie tabel voor de LLM
        summary = subset.groupby(['Finny_Year', 'Finny_GLDescription'])['AmountDC_num'].sum().reset_index()
        data_str = f"Gevonden: {len(subset)} transacties.\nTOTAAL: € {total:,.2f}\n\nDetails:\n{summary.to_string()}"
        return data_str, report
    else:
        return "Geen resultaten.", report

# ==========================================
# 4. MAIN APP
# ==========================================
def check_password():
    """Simpele login"""
    if "auth" not in st.session_state: st.session_state.auth = False
    if not st.session_state.auth:
        pwd = st.text_input("Wachtwoord", type="password")
        if pwd == "demo2025": st.session_state.auth = True; st.rerun()
        return False
    return True

if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # SIDEBAR
    with st.sidebar:
        # Logo Check
        if os.path.exists("finny_logo.jpg"):
            st.image("finny_logo.jpg", width=120)
        else:
            st.warning("Logo niet gevonden (upload finny_logo.jpg)")
            
        st.markdown("### ⚙️ Debug Console")
        data = load_data()
        
        if data["trans"] is not None:
            st.success(f"CSV: {len(data['trans'])} rijen")
            st.caption(f"Jaren: {data['years']}")
        
        if st.button("🧹 Cache Legen"):
            st.cache_data.clear()
            st.rerun()

    # CHAT
    st.title("🦁 Finny 6.0 | Glass Box")
    st.markdown("Stel je vraag. Ik laat onderaan precies zien hoe ik zoek.")

    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: 
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if "debug" in m:
                with st.expander("🔎 Analyse van de zoekopdracht"):
                    for line in m["debug"]: st.text(line)

    if prompt := st.chat_input("Bijv: Wat waren de autokosten in 2023?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            # 1. PLAN MAKEN
            plan = get_search_plan(client, prompt, data["syllabus"], data["years"])
            
            # 2. ZOEKEN (met rapport)
            csv_result, report = execute_search_with_report(plan, data["trans"])
            
            # 3. ANTWOORD GENEREREN
            full_context = f"CSV DATA:\n{csv_result}\n\nPDF DATA (indien relevant):\n{data['pdf_text'][:2000]}"
            
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Je bent Finny. Antwoord kort en bondig op basis van de data. Als er CSV data is: geef het bedrag. Als er geen data is: zeg dat eerlijk."},
                    {"role": "user", "content": f"Vraag: {prompt}\n\nData:\n{full_context}"}
                ]
            )
            ans = completion.choices[0].message.content
            
            st.markdown(ans)
            
            # 4. DEBUG TONEN
            with st.expander("🔎 Analyse van de zoekopdracht", expanded=True):
                st.markdown(f"**Strategie:** Ik zocht naar GL Codes `{plan.get('gl_codes')}` in jaren `{plan.get('years')}`.")
                st.markdown("**Logboek:**")
                for line in report:
                    st.text(line)
            
            # Opslaan in history
            st.session_state.messages.append({"role": "assistant", "content": ans, "debug": report})
