import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
from datetime import datetime
import re

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
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal (v6.0 Memory)</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# ==========================================
# 2. DATA ENGINE (MET GEHEUGEN & RGS)
# ==========================================

@st.cache_data
def load_knowledge_base():
    content = ""
    syllabus_extract = "" 
    
    # Profiel
    if os.path.exists("van_hattem_advies_profiel.txt"):
        try:
            with open("van_hattem_advies_profiel.txt", "r", encoding="utf-8") as f:
                content += f"--- KLANTPROFIEL ---\n{f.read()}\n\n"
        except: pass

    # Syllabus + RGS Schema (Cruciaal voor synoniemen)
    if os.path.exists("Finny_syllabus.txt"):
        try:
            with open("Finny_syllabus.txt", "r", encoding="utf-8") as f:
                full = f.read()
                content += f"--- SYLLABUS & RGS ---\n{full}\n\n"
                syllabus_extract = full[:4000] # Meer context voor de vertaler
        except: pass
        
    # Extra: Laad RGS CSV als die er is voor de vertaler
    if os.path.exists("133700 Standaard Rekeningschema Template FinGLAccountSearch.csv"):
        try:
            rgs_df = pd.read_csv("133700 Standaard Rekeningschema Template FinGLAccountSearch.csv", sep="\t", on_bad_lines='skip')
            # Voeg de rekeningschema namen toe aan de syllabus context
            rgs_text = rgs_df.to_string(index=False)
            syllabus_extract += f"\n\n--- RGS REKENINGSCHEMA OPTIES ---\n{rgs_text[:4000]}"
        except: pass
        
    return content, syllabus_extract

def execute_smart_query(intent, full_df):
    """
    Voert de zoekopdracht uit.
    """
    if full_df is None: return "--- GEEN DATA ---"
    if intent.get('source') == 'PDF': return "" 
    
    try:
        df = full_df.copy()
        
        # 1. Datum Filter
        years = intent.get('years', [])
        # Als geen jaar, pak dan ALLES (veiliger voor demo)
        date_col = next((c for c in df.columns if 'datum' in c.lower() or 'date' in c.lower()), None)
        
        if date_col and years:
            df['dt_temp'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            # Filter op jaar
            df_year = df[df['dt_temp'].dt.year.isin(years)]
            if len(df_year) > 0:
                df = df_year
            else:
                # FALLBACK: Als jaar 0 resultaten geeft, negeer jaarfilter
                pass 
        
        # 2. Tekst Filter (Synoniemen)
        terms = intent.get('search_terms', [])
        if terms:
            text_cols = [c for c in df.columns if any(x in c.lower() for x in ['omschrijving', 'desc', 'naam', 'name', 'relatie', 'grootboek', 'dim1'])]
            if text_cols:
                # Maak veilige regex
                safe_terms = [re.escape(t) for t in terms]
                pattern = '|'.join(safe_terms)
                mask = df[text_cols].astype(str).agg(' '.join, axis=1).str.contains(pattern, case=False, na=False)
                df = df[mask]
        
        # 3. Resultaat
        if len(df) > 0:
            amount_col = next((c for c in df.columns if 'bedrag' in c.lower() or 'value' in c.lower()), None)
            totaal = 0.0
            if amount_col: totaal = df[amount_col].sum()
            
            # Format
            if len(df) <= 50:
                cols = [c for c in df.columns if c not in ['dt_temp']]
                return f"""
                --- CSV RESULTAAT (Gefilterd op {terms}) ---
                Aantal transacties: {len(df)}
                GEVERIFIEERD TOTAALBEDRAG: € {totaal:.2f}
                
                Details:
                {df[cols].to_string(index=False)}
                """
            else:
                desc_col = next((c for c in df.columns if 'omschrijving' in c.lower()), df.columns[0])
                summary = df.groupby(desc_col)[amount_col].sum().sort_values().head(20).reset_index()
                return f"""
                --- CSV SAMENVATTING (Gefilterd op {terms}) ---
                Aantal transacties: {len(df)}
                GEVERIFIEERD TOTAALBEDRAG: € {totaal:.2f}
                
                Top 20 Posten:
                {summary.to_string(index=False)}
                """
                
        return f"--- GEEN TRANSACTIES GEVONDEN VOOR: {terms} ---"

    except Exception as e:
        return f"--- ERROR CSV: {e} ---"

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
    if not found and intent.get('source') == 'PDF': return "--- GEEN JAARREKENINGEN GEVONDEN ---"
    return content

# ==========================================
# 3. INTELLIGENTIE MET GEHEUGEN
# ==========================================

def clean_json(text):
    return text.replace("```json", "").replace("```", "").strip()

def get_conversation_history():
    """Haalt de laatste 4 berichten op voor context"""
    history = []
    if "messages" in st.session_state:
        # Pak de laatste 4 berichten (anders wordt de prompt te groot)
        recent = st.session_state.messages[-4:]
        for msg in recent:
            history.append({"role": msg["role"], "content": msg["content"]})
    return history

def run_finny_mini(client, question, syllabus):
    """
    De Strateeg. Vertaalt vraag naar RGS codes.
    """
    # We sturen de historie mee zodat hij "en vorig jaar?" snapt
    history = get_conversation_history()
    
    system_prompt = f"""
    Je bent Finny-Mini (v9.7). Query Translator.
    Datum: {datetime.now().strftime("%Y-%m-%d")}
    
    Kennis (RGS & Syllabus): 
    {syllabus}
    
    OPDRACHT:
    Vertaal de laatste gebruikersvraag naar een database-query.
    
    REGELS VOOR SYNONIEMEN:
    - Zoek in de RGS lijst naar termen die matchen.
    - "Auto" -> ["Brandstof", "Wegenbelasting", "Garage", "4100", "4200"] (Kijk naar RGS)
    - "Kantoor" -> ["Huur", "Kantoorkosten", "Inventaris", "4300"]
    - "Telefoon" -> ["KPN", "Vodafone", "T-Mobile", "Telefoonkosten", "Internet"]
    
    REGELS VOOR GEHEUGEN:
    - Kijk naar de voorgaande vragen. Als de gebruiker zegt "En in 2022?", gebruik dan het onderwerp van de vorige vraag.
    
    Output JSON:
    {{
        "source": "CSV" | "PDF" | "BOTH",
        "years": [int],
        "search_terms": [strings],
        "reasoning": "string"
    }}
    """
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]
    
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(clean_json(res.choices[0].message.content))
    except:
        return {"source": "BOTH", "years": [], "search_terms": [], "reasoning": "Error"}

def run_finny_main(client, question, context):
    """
    De Antwoorder. Met geheugen.
    """
    history = get_conversation_history()
    
    system_prompt = f"""
    Je bent Finny (v9.9). Financiële AI.
    
    DATA CONTEXT (Hierin staat het antwoord):
    {context}
    
    Instructies:
    1. Gebruik GEVERIFIEERDE TOTALEN uit de tekst. Reken niet zelf.
    2. Als de data 0 is of verschilt, leg uit wat je ziet in de data.
    3. Onthoud de context van het gesprek.
    """
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]
    
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
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
        if csv_df is not None: st.success("✅ Twinfield Live (v6.0)")
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

        with st.chat_message("assistant"):
            with st.spinner("Analyseren..."):
                # STAP 1: MINI (Met RGS Kennis & Geheugen)
                intent = run_finny_mini(client, prompt, syllabus_extract)
                
                with st.expander("🕵️ Debug Finny", expanded=False):
                    st.write(intent)

                # STAP 2: DATA
                context_data = static_context
                source = intent.get('source', 'BOTH')
                
                if source in ['CSV', 'BOTH']:
                    context_data += execute_smart_query(intent, csv_df)
                
                if source in ['PDF', 'BOTH']:
                    context_data += get_pdf_context(intent)

                # STAP 3: ANTWOORD (Met Geheugen)
                response = run_finny_main(client, prompt, context_data)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
