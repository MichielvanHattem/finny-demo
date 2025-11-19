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
    """Beveiligde toegang."""
    def password_entered():
        if st.session_state["password"] == "demo2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Logo check (veilig)
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal (v5.3 Stable)</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# ==========================================
# 2. DATA LAAG (ROBUUST)
# ==========================================

@st.cache_data
def load_knowledge_base():
    """
    Laadt de statische kennis. 
    FIX: Variabelen worden nu vooraf geïnitialiseerd om UnboundLocalError te voorkomen.
    """
    content = ""
    syllabus_extract = "" 
    
    # 1. Profiel laden
    if os.path.exists("van_hattem_advies_profiel.txt"):
        try:
            with open("van_hattem_advies_profiel.txt", "r", encoding="utf-8") as f:
                content += f"--- KLANTPROFIEL ---\n{f.read()}\n\n"
        except Exception as e:
            print(f"Fout bij laden profiel: {e}")
            
    # 2. Syllabus laden
    if os.path.exists("Finny_syllabus.txt"):
        try:
            with open("Finny_syllabus.txt", "r", encoding="utf-8") as f:
                full_text = f.read()
                content += f"--- SYLLABUS & RGS ---\n{full_text}\n\n"
                syllabus_extract = full_text[:3000] # Voor de Mini prompt
        except Exception as e:
            print(f"Fout bij laden syllabus: {e}")
            
    return content, syllabus_extract

def execute_smart_query(intent, full_df):
    """
    Voert de zoekopdracht uit in Python om token-overflow en rekenfouten te voorkomen.
    """
    if full_df is None: return ""
    if intent.get('source') == 'PDF': return "" # Skip CSV logica als PDF gevraagd is
    
    try:
        df = full_df.copy()
        
        # A. Datum Filter
        years = intent.get('years', [])
        # Als geen jaar gevonden is, en bron is CSV, pakken we veiligheidshalve het huidige en vorige jaar
        if not years: 
            curr_year = datetime.now().year
            years = [curr_year, curr_year - 1]
        
        date_col = next((c for c in df.columns if 'datum' in c.lower() or 'date' in c.lower()), None)
        if date_col:
            df['dt_temp'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df[df['dt_temp'].dt.year.isin(years)]
        
        # B. Tekst/RGS Filter (Finny-Mini 9.6 Logic)
        terms = intent.get('search_terms', [])
        if terms:
            text_cols = [c for c in df.columns if any(x in c.lower() for x in ['omschrijving', 'desc', 'naam', 'name', 'relatie', 'grootboek', 'dim1'])]
            if text_cols:
                # Maak een veilige string kolom voor de zoekactie
                pattern = '|'.join([re.escape(term) for term in terms]) # Escape voorkomt regex errors
                mask = df[text_cols].astype(str).agg(' '.join, axis=1).str.contains(pattern, case=False, na=False)
                df = df[mask]
                
        # C. Aggregatie
        if len(df) > 0:
            amount_col = next((c for c in df.columns if 'bedrag' in c.lower() or 'value' in c.lower()), None)
            totaal_bedrag = 0.0
            if amount_col:
                totaal_bedrag = df[amount_col].sum()
            
            # Korte lijst: Details tonen
            if len(df) <= 50:
                cols = [c for c in df.columns if c not in ['dt_temp']]
                return f"""
                --- CSV RESULTATEN (Gefilterd op {terms} in {years}) ---
                Aantal transacties: {len(df)}
                GEVERIFIEERD TOTAALBEDRAG: € {totaal_bedrag:.2f}
                
                Details:
                {df[cols].to_string(index=False)}
                """
            else:
                # Lange lijst: Samenvatten
                desc_col = next((c for c in df.columns if 'omschrijving' in c.lower()), df.columns[0])
                summary = df.groupby(desc_col)[amount_col].sum().sort_values().head(20).reset_index()
                return f"""
                --- CSV SAMENVATTING (Gefilterd op {terms} in {years}) ---
                Aantal transacties: {len(df)} (Te veel voor details, hier is de top 20)
                GEVERIFIEERD TOTAALBEDRAG: € {totaal_bedrag:.2f}
                
                Top 20 Posten:
                {summary.to_string(index=False)}
                """
                
        return f"--- GEEN TRANSACTIES GEVONDEN VOOR: {terms} IN {years} ---\n"
    
    except Exception as e:
        return f"--- FOUT BIJ ANALYSEREN CSV: {e} ---"

def get_pdf_context(intent):
    """Haalt PDF data op. Faalt stil (geen crash) als bestanden missen."""
    if intent.get('source') == 'CSV': return "" 
    
    content = ""
    pdfs = [
        "Van Hattem Advies B.V. - Jaarrekening 2024.pdf", 
        "Van Hattem Advies B.V. - Jaarrekening 2023.pdf", 
        "Van Hattem Advies B.V. - Jaarstukken 2022.pdf"
    ]
    found = False
    for pdf in pdfs:
        if os.path.exists(pdf):
            try:
                reader = PdfReader(pdf)
                text = ""
                # Scan eerste 15 pagina's voor balans/W&V
                for i, page in enumerate(reader.pages):
                    if i < 15: text += page.extract_text()
                content += f"--- BRON: {pdf} ---\n{text[:6000]}\n\n"
                found = True
            except: pass
            
    if not found and intent.get('source') == 'PDF':
        return "--- LET OP: Geen jaarrekeningen gevonden op het systeem. ---"
    return content

# ==========================================
# 3. INTELLIGENTIE LAAG (PROMPTS)
# ==========================================

def clean_json_response(response_text):
    """
    Veiligheidsfunctie: Soms zet GPT ```json ... ``` om de output. 
    Dit haalt dat weg om crashes te voorkomen.
    """
    clean_text = response_text.replace("```json", "").replace("```", "").strip()
    return clean_text

def run_finny_mini(client, question, syllabus_extract):
    """
    FINNY-MINI (Versie 9.6/9.7)
    Vertaalt vragen naar zoektermen en RGS codes.
    """
    system_prompt = f"""
    Je bent Finny-Mini. Je bent een query-translator, GEEN chatbot.
    
    HUIDIGE DATUM: {datetime.now().strftime("%Y-%m-%d")}
    
    KENNIS (Syllabus/RGS):
    {syllabus_extract}
    
    OPDRACHT:
    1. Analyseer de intentie: CSV (Details/Geld) of PDF (Inzicht/Balans)?
    2. Vertaal de vraag naar concrete zoektermen. Denk aan RGS codes als die in de syllabus staan.
       Voorbeeld: "Auto" -> ["Brandstof", "Garage", "4100", "Wegenbelasting"]
    3. Bepaal jaren. "Vorig jaar" is rekenwerk t.o.v. huidige datum.
    
    OUTPUT (JSON ONLY):
    {{
        "source": "CSV" | "PDF" | "BOTH",
        "years": [int],
        "search_terms": [strings],
        "reasoning": "string"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        cleaned_json = clean_json_response(response.choices[0].message.content)
        return json.loads(cleaned_json)
    except Exception as e:
        # Fallback strategie als Mini faalt
        print(f"Mini Error: {e}")
        return {"source": "BOTH", "years": [], "search_terms": []}

def run_finny_main(client, question, context):
    """
    FINNY MAIN (Versie 9.9)
    Het gezicht naar de klant.
    """
    system_prompt = f"""
    Je bent Finny, de financiële AI-partner.
    
    DATA CONTEXT:
    {context}
    
    REGELS (PROMPT 9.9):
    1. Conclusie Eerst: Geef direct antwoord.
    2. Feiten: Gebruik de 'GEVERIFIEERD TOTAALBEDRAG' cijfers uit de context. Ga NIET zelf optellen.
    3. Bronvermelding: Zeg waar je het vandaan hebt.
    4. Opmaak: Markdown tabellen.
    5. Geen data? Zeg het eerlijk.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            temperature=0
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Fout bij genereren antwoord: {e}"

# ==========================================
# 4. MAIN APP LOOP
# ==========================================

if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt. Stel deze in bij 'Secrets'.")
        st.stop()

    # Data Laden (Eénmalig, veilig)
    csv_df = None
    if os.path.exists("133700 FinTransactionSearch all 5jr.csv"):
        try:
            csv_df = pd.read_csv("133700 FinTransactionSearch all 5jr.csv", sep=";", on_bad_lines='skip', low_memory=False)
        except Exception as e:
             st.error(f"Fout bij laden CSV: {e}")
    
    static_context, syllabus_extract = load_knowledge_base()

    # Sidebar
    with st.sidebar:
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("### 🏢 Van Hattem Advies B.V.")
        if csv_df is not None: st.success("✅ Twinfield Live (v5.3)")
        else: st.error("❌ Data Connectie Fout")
        st.markdown("---")
        if st.button("Reset"): st.rerun()

    st.title("👋 Goedemiddag, Michiel.")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # STAP 1: FINNY-MINI (Intentie)
        with st.chat_message("assistant"):
            with st.spinner("Finny vertaalt vraag..."):
                intent = run_finny_mini(client, prompt, syllabus_extract)
                
                # Debug info (SaaS ervaring)
                debug_msg = f"🧠 **Strategie:** {intent.get('source', 'Onbekend')}"
                if intent.get('years'): debug_msg += f" | Jaren: {intent['years']}"
                if intent.get('search_terms'): debug_msg += f" | Zoektermen: {intent['search_terms']}"
                st.info(debug_msg)

            # STAP 2: DATA OPHALEN (Python Engine)
            with st.spinner("Data verzamelen..."):
                # Altijd profiel/syllabus als basis
                context_data = static_context
                
                source = intent.get('source', 'BOTH')
                
                if source in ['CSV', 'BOTH']:
                    context_data += execute_smart_query(intent, csv_df)
                
                if source in ['PDF', 'BOTH']:
                    context_data += get_pdf_context(intent)

            # STAP 3: ANTWOORD (Finny Main)
            with st.spinner("Finny formuleert antwoord..."):
                response = run_finny_main(client, prompt, context_data)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
