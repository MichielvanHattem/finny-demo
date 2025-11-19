import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
from datetime import datetime
import re

# ==========================================
# 1. CONFIGURATIE & AUTHENTICATIE
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
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal (v5.4 Auto-Fallback)</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# ==========================================
# 2. DATA LAAG
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
                full_text = f.read()
                content += f"--- SYLLABUS & RGS ---\n{full_text}\n\n"
                syllabus_extract = full_text[:3000]
        except: pass
            
    return content, syllabus_extract

def execute_smart_query(intent, full_df):
    """
    Slimme zoekfunctie met 'Auto-Fallback'.
    Als zoeken in 2024 niks oplevert, zoekt hij automatisch in ALLE jaren.
    """
    if full_df is None: return ""
    if intent.get('source') == 'PDF': return "" 
    
    try:
        df = full_df.copy()
        
        # 1. Datum Conversie (Verbeterd voor NL CSVs: dayfirst=True)
        date_col = next((c for c in df.columns if 'datum' in c.lower() or 'date' in c.lower()), None)
        if date_col:
            # dayfirst=True is cruciaal voor Nederlandse 01-02-2023 formaten!
            df['dt_temp'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        else:
            df['dt_temp'] = pd.NaT # Geen datum? Dan werkt datumfilter niet, maar tekstfilter wel.

        # 2. Zoektermen Filter (Altijd toepassen)
        terms = intent.get('search_terms', [])
        filtered_df = df.copy()
        
        if terms:
            text_cols = [c for c in df.columns if any(x in c.lower() for x in ['omschrijving', 'desc', 'naam', 'name', 'relatie', 'grootboek', 'dim1'])]
            if text_cols:
                pattern = '|'.join([re.escape(term) for term in terms])
                mask = df[text_cols].astype(str).agg(' '.join, axis=1).str.contains(pattern, case=False, na=False)
                filtered_df = df[mask]

        # 3. Jaar Filter (MET FALLBACK LOGICA)
        years = intent.get('years', [])
        final_df = pd.DataFrame()
        used_years_msg = ""

        if years and date_col:
            # Probeer eerst te filteren op de gevraagde jaren
            final_df = filtered_df[filtered_df['dt_temp'].dt.year.isin(years)]
            
            # *** DE HACK: Als we niks vinden, negeer het jaarfilter! ***
            if len(final_df) == 0:
                final_df = filtered_df # Pak alles wat matcht op tekst
                if len(final_df) > 0:
                    used_years_msg = "(Geen resultaat in gevraagde jaren, gezocht in hele historie)"
            else:
                used_years_msg = f"(Gefilterd op jaren: {years})"
        else:
            final_df = filtered_df
            used_years_msg = "(Alle jaren)"

        # 4. Resultaat opbouwen
        if len(final_df) > 0:
            amount_col = next((c for c in final_df.columns if 'bedrag' in c.lower() or 'value' in c.lower()), None)
            totaal_bedrag = 0.0
            if amount_col:
                totaal_bedrag = final_df[amount_col].sum()
            
            if len(final_df) <= 50:
                cols = [c for c in final_df.columns if c not in ['dt_temp']]
                return f"""
                --- CSV RESULTATEN {used_years_msg} ---
                Zoektermen: {terms}
                Aantal transacties: {len(final_df)}
                GEVERIFIEERD TOTAALBEDRAG: € {totaal_bedrag:.2f}
                
                Details:
                {final_df[cols].to_string(index=False)}
                """
            else:
                desc_col = next((c for c in final_df.columns if 'omschrijving' in c.lower()), final_df.columns[0])
                summary = final_df.groupby(desc_col)[amount_col].sum().sort_values().head(20).reset_index()
                return f"""
                --- CSV SAMENVATTING {used_years_msg} ---
                Zoektermen: {terms}
                Aantal transacties: {len(final_df)}
                GEVERIFIEERD TOTAALBEDRAG: € {totaal_bedrag:.2f}
                
                Top 20 Posten:
                {summary.to_string(index=False)}
                """
                
        return f"--- GEEN TRANSACTIES GEVONDEN VOOR: {terms} --- (Ook niet na zoeken in alle jaren)\n"
    
    except Exception as e:
        return f"--- FOUT BIJ ANALYSEREN CSV: {e} ---"

def get_pdf_context(intent):
    if intent.get('source') == 'CSV': return "" 
    content = ""
    pdfs = ["Van Hattem Advies B.V. - Jaarrekening 2024.pdf", "Van Hattem Advies B.V. - Jaarrekening 2023.pdf", "Van Hattem Advies B.V. - Jaarstukken 2022.pdf"]
    found = False
    for pdf in pdfs:
        if os.path.exists(pdf):
            try:
                reader = PdfReader(pdf)
                text = ""
                for i, page in enumerate(reader.pages):
                    if i < 15: text += page.extract_text()
                content += f"--- BRON: {pdf} ---\n{text[:6000]}\n\n"
                found = True
            except: pass
    if not found and intent.get('source') == 'PDF': return "--- LET OP: Geen jaarrekeningen gevonden. ---"
    return content

# ==========================================
# 3. INTELLIGENTIE (PROMPTS)
# ==========================================

def clean_json_response(response_text):
    return response_text.replace("```json", "").replace("```", "").strip()

def run_finny_mini(client, question, syllabus_extract):
    system_prompt = f"""
    Je bent Finny-Mini (v9.7). Query Translator.
    Huidige datum: {datetime.now().strftime("%Y-%m-%d")}
    
    Kennis: {syllabus_extract}
    
    Opdracht:
    1. Vertaal vraag naar zoektermen (RGS/Synoniemen).
    2. Bepaal jaren.
    3. Kies bron: CSV (Transacties) of PDF (Jaarrekening).
    
    Output JSON only.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(clean_json_response(response.choices[0].message.content))
    except:
        return {"source": "BOTH", "years": [], "search_terms": []}

def run_finny_main(client, question, context):
    system_prompt = f"""
    Je bent Finny (v9.9).
    
    Data Context:
    {context}
    
    Instructies:
    1. Gebruik de GEVERIFIEERDE TOTALEN uit de CSV-tekst. Reken niet zelf.
    2. Als er staat "(Geen resultaat in gevraagde jaren, gezocht in hele historie)", meld dit dan aan de gebruiker ("Ik heb gekeken in alle jaren omdat er in 2024 geen data was").
    3. Wees zakelijk en direct.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            temperature=0
        )
        return completion.choices[0].message.content
    except Exception as e: return str(e)

# ==========================================
# 4. APP LOOP
# ==========================================

if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt")
        st.stop()

    csv_df = None
    if os.path.exists("133700 FinTransactionSearch all 5jr.csv"):
        try:
            # dayfirst=True is belangrijk voor NL data!
            csv_df = pd.read_csv("133700 FinTransactionSearch all 5jr.csv", sep=";", on_bad_lines='skip', low_memory=False)
        except: pass
    
    static_context, syllabus_extract = load_knowledge_base()

    with st.sidebar:
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("### 🏢 Van Hattem Advies B.V.")
        if csv_df is not None: st.success("✅ Twinfield Live (v5.4)")
        else: st.error("❌ Data Connectie Fout")
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

        with st.chat_message("assistant"):
            with st.spinner("Analyseren..."):
                intent = run_finny_mini(client, prompt, syllabus_extract)
                
                # Debug info
                debug_msg = f"🧠 **Strategie:** {intent.get('source')} | Zoekt: {intent.get('search_terms')}"
                st.info(debug_msg)

                context_data = static_context
                source = intent.get('source', 'BOTH')
                
                if source in ['CSV', 'BOTH']:
                    context_data += execute_smart_query(intent, csv_df)
                
                if source in ['PDF', 'BOTH']:
                    context_data += get_pdf_context(intent)

                response = run_finny_main(client, prompt, context_data)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
