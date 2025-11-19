import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
from datetime import datetime
import re  # <--- HIER ZAT DE FOUT! DEZE WAS NODIG.

# ==========================================
# 1. CONFIGURATIE
# ==========================================

st.set_page_config(page_title="Finny | Intelligent Finance", page_icon="💰", layout="wide")

def check_password():
    def password_entered():
        if st.session_state["password"] == "demo2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal (v5.5 Debug)</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# ==========================================
# 2. DATA FUNCTIES
# ==========================================

@st.cache_data
def load_knowledge_base():
    content = ""
    syllabus_extract = "" 
    if os.path.exists("van_hattem_advies_profiel.txt"):
        try:
            with open("van_hattem_advies_profiel.txt", "r", encoding="utf-8") as f:
                content += f"--- KLANTPROFIEL ---\n{f.read()}\n\n"
        except: pass
    if os.path.exists("Finny_syllabus.txt"):
        try:
            with open("Finny_syllabus.txt", "r", encoding="utf-8") as f:
                full = f.read()
                content += f"--- SYLLABUS ---\n{full}\n\n"
                syllabus_extract = full[:3000]
        except: pass
    return content, syllabus_extract

def execute_smart_query(intent, full_df):
    """
    Zoekt in CSV. Nu met 're' import gefixt, dus crasht niet meer.
    """
    if full_df is None: return "--- GEEN CSV BESTAND GELADEN ---"
    if intent.get('source') == 'PDF': return "" 
    
    try:
        df = full_df.copy()
        
        # 1. Datum Filter
        # Fallback: Als Finny geen jaren geeft, pakken we alles.
        years = intent.get('years', [])
        date_col = next((c for c in df.columns if 'datum' in c.lower() or 'date' in c.lower()), None)
        
        if date_col:
            df['dt_temp'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            if years:
                # Filter op jaar
                df_year = df[df['dt_temp'].dt.year.isin(years)]
                # Als jaarfilter resultaten oplevert, gebruik die. Anders val terug op alles (Fallback)
                if len(df_year) > 0:
                    df = df_year
        
        # 2. Tekst Filter (Cruciaal voor kosten vragen)
        terms = intent.get('search_terms', [])
        if terms:
            # Zoek in alle tekst-achtige kolommen
            text_cols = [c for c in df.columns if any(x in c.lower() for x in ['omschrijving', 'desc', 'naam', 'name', 'relatie', 'grootboek'])]
            if text_cols:
                # Maak veilige regex (voorkomt crash op leestekens)
                pattern = '|'.join([re.escape(t) for t in terms])
                mask = df[text_cols].astype(str).agg(' '.join, axis=1).str.contains(pattern, case=False, na=False)
                df = df[mask]
        
        # 3. Resultaat
        if len(df) > 0:
            amount_col = next((c for c in df.columns if 'bedrag' in c.lower() or 'value' in c.lower()), None)
            totaal = 0.0
            if amount_col: totaal = df[amount_col].sum()
            
            # Formatteren
            if len(df) <= 50:
                cols = [c for c in df.columns if c not in ['dt_temp']]
                return f"""
                --- CSV RESULTAAT ---
                Zoektermen: {terms}
                Jaren filter: {years if years else "Alle jaren"}
                Aantal transacties: {len(df)}
                GEVERIFIEERD TOTAALBEDRAG: € {totaal:.2f}
                
                Details:
                {df[cols].to_string(index=False)}
                """
            else:
                desc_col = next((c for c in df.columns if 'omschrijving' in c.lower()), df.columns[0])
                summary = df.groupby(desc_col)[amount_col].sum().sort_values().head(20).reset_index()
                return f"""
                --- CSV SAMENVATTING ---
                Zoektermen: {terms}
                Aantal transacties: {len(df)}
                GEVERIFIEERD TOTAALBEDRAG: € {totaal:.2f}
                
                Top 20:
                {summary.to_string(index=False)}
                """
                
        return f"--- GEEN TRANSACTIES GEVONDEN VOOR: {terms} ---"

    except Exception as e:
        return f"--- ERROR IN CSV QUERY: {e} ---"

def get_pdf_context(intent):
    if intent.get('source') == 'CSV': return ""
    content = ""
    pdfs = ["Van Hattem Advies B.V. - Jaarrekening 2024.pdf", "Van Hattem Advies B.V. - Jaarrekening 2023.pdf", "Van Hattem Advies B.V. - Jaarstukken 2022.pdf"]
    for pdf in pdfs:
        if os.path.exists(pdf):
            try:
                reader = PdfReader(pdf)
                text = ""
                for i, page in enumerate(reader.pages):
                    if i < 15: text += page.extract_text()
                content += f"--- BRON: {pdf} ---\n{text[:6000]}\n\n"
            except: pass
    return content

# ==========================================
# 3. PROMPTS
# ==========================================

def clean_json(text):
    return text.replace("```json", "").replace("```", "").strip()

def run_finny_mini(client, question, syllabus):
    """
    De Strateeg. Vertaalt vraag naar zoekopdracht.
    """
    prompt = f"""
    Je bent Finny-Mini (v9.7). Query Translator.
    Datum: {datetime.now().strftime("%Y-%m-%d")}
    
    Kennis: {syllabus}
    
    Opdracht: Vertaal vraag naar RGS-zoektermen.
    Voorbeeld: "Autokosten" -> ["Brandstof", "Garage", "Wegenbelasting", "4100"]
    
    Output JSON only:
    {{
        "source": "CSV" | "PDF" | "BOTH",
        "years": [int],
        "search_terms": [strings],
        "reasoning": "string"
    }}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(clean_json(res.choices[0].message.content))
    except:
        return {"source": "BOTH", "years": [], "search_terms": [], "reasoning": "Error in Mini"}

def run_finny_main(client, question, context):
    prompt = f"""
    Je bent Finny.
    Data Context:
    {context}
    
    Regels:
    1. Gebruik GEVERIFIEERDE TOTALEN uit de CSV-tekst. Reken niet zelf.
    2. Als CSV leeg is, kijk in PDF.
    3. Geef direct antwoord.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}],
            temperature=0
        )
        return res.choices[0].message.content
    except Exception as e: return str(e)

# ==========================================
# 4. APP
# ==========================================

if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt")
        st.stop()

    # Laad CSV
    csv_df = None
    if os.path.exists("133700 FinTransactionSearch all 5jr.csv"):
        try:
            csv_df = pd.read_csv("133700 FinTransactionSearch all 5jr.csv", sep=";", on_bad_lines='skip', low_memory=False)
        except: pass
    
    static_context, syllabus_extract = load_knowledge_base()

    with st.sidebar:
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("### 🏢 Van Hattem Advies B.V.")
        if csv_df is not None: st.success("✅ Twinfield Live")
        else: st.error("❌ Data Fout")
        st.markdown("---")
        if st.button("Reset"): st.rerun()

    st.title("👋 Goedemiddag, Michiel.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # STAP 1: MINI (Strategie)
        with st.chat_message("assistant"):
            with st.spinner("Analyseren..."):
                intent = run_finny_mini(client, prompt, syllabus_extract)
                
                # --- HIER IS DE VISUELE DEBUGGER VOOR JOU ---
                with st.expander("🕵️ Finny's Denkproces (Debug)", expanded=True):
                    st.write(f"**Strategie:** {intent.get('reasoning', 'Geen uitleg')}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Bron", intent.get('source'))
                    col2.metric("Jaren", str(intent.get('years')))
                    col3.metric("Zoektermen", str(intent.get('search_terms')))
                # --------------------------------------------

                context_data = static_context
                source = intent.get('source', 'BOTH')
                
                if source in ['CSV', 'BOTH']:
                    csv_result = execute_smart_query(intent, csv_df)
                    context_data += csv_result
                    # Debug: Laat zien als CSV faalt
                    if "ERROR" in csv_result or "GEEN TRANSACTIES" in csv_result:
                        st.caption(f"⚠️ CSV Resultaat: {csv_result.strip()[:100]}...")
                
                if source in ['PDF', 'BOTH']:
                    context_data += get_pdf_context(intent)

                response = run_finny_main(client, prompt, context_data)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
