import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
from datetime import datetime

# ==========================================
# 1. CONFIGURATIE & AUTHENTICATIE
# ==========================================

st.set_page_config(page_title="Finny | Intelligent Finance", page_icon="💰", layout="wide")

def check_password():
    """Beveiligde toegang zoals vereist in architectuur."""
    def password_entered():
        if st.session_state["password"] == "demo2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Logo check
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal (v5.2 Stable)</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# ==========================================
# 2. DATA LAAG (PYTHON ENGINE)
# ==========================================

@st.cache_data
def load_knowledge_base():
    """Laadt de statische kennis (Syllabus/Profiel) voor RGS mapping."""
    content = ""
    syllabus_extract = ""
    
    # Profiel
    if os.path.exists("van_hattem_advies_profiel.txt"):
        with open("van_hattem_advies_profiel.txt", "r", encoding="utf-8") as f:
            content += f"--- KLANTPROFIEL ---\n{f.read()}\n\n"
            
    # Syllabus (Cruciaal voor Finny-Mini 9.6)
    if os.path.exists("Finny_syllabus.txt"):
        with open("Finny_syllabus.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
            content += f"--- SYLLABUS & RGS ---\n{full_text}\n\n"
            syllabus_extract = full_text[:3000] # Voor de Mini prompt
            
    return content, syllabus_extract

def execute_smart_query(intent, full_df):
    """
    De 'Execution Engine'. Voert de zoekopdracht uit en rekent totalen uit.
    Dit voorkomt hallucinaties (Les uit Testverslag 9.3).
    """
    if full_df is None: return ""
    if intent['source'] == 'PDF': return "" # Alleen CSV logica hier
    
    df = full_df.copy()
    
    # A. Datum Filter (Standaard jaar 2024/2025 als niks gevonden)
    years = intent.get('years', [])
    if not years: years = [datetime.now().year]
    
    date_col = next((c for c in df.columns if 'datum' in c.lower() or 'date' in c.lower()), None)
    if date_col:
        df['dt_temp'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['dt_temp'].dt.year.isin(years)]
    
    # B. Tekst/RGS Filter (Finny-Mini 9.6 Logic)
    terms = intent.get('search_terms', [])
    if terms:
        # We zoeken in Omschrijving, Relatie EN Grootboek (voor RGS codes)
        text_cols = [c for c in df.columns if any(x in c.lower() for x in ['omschrijving', 'desc', 'naam', 'name', 'relatie', 'grootboek', 'dim1'])]
        if text_cols:
            pattern = '|'.join(terms)
            mask = df[text_cols].astype(str).agg(' '.join, axis=1).str.contains(pattern, case=False, na=False)
            df = df[mask]
            
    # C. Aggregatie (De 'Rekenmachine' fix)
    # Als we resultaten hebben, rekenen we direct het totaal uit.
    if len(df) > 0:
        amount_col = next((c for c in df.columns if 'bedrag' in c.lower() or 'value' in c.lower()), None)
        totaal_bedrag = 0.0
        if amount_col:
            totaal_bedrag = df[amount_col].sum()
        
        # Als lijst kort is: Toon details
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
            # Als lijst lang is: Toon samenvatting per post
            desc_col = next((c for c in df.columns if 'omschrijving' in c.lower()), df.columns[0])
            summary = df.groupby(desc_col)[amount_col].sum().sort_values().head(20).reset_index()
            return f"""
            --- CSV SAMENVATTING (Gefilterd op {terms} in {years}) ---
            Aantal transacties: {len(df)} (Te veel voor details)
            GEVERIFIEERD TOTAALBEDRAG: € {totaal_bedrag:.2f}
            
            Top 20 Posten:
            {summary.to_string(index=False)}
            """
            
    return f"--- GEEN TRANSACTIES GEVONDEN VOOR: {terms} IN {years} ---\n"

def get_pdf_context(intent):
    """Haalt PDF data op als intentie daarom vraagt."""
    if intent['source'] == 'CSV': return "" 
    
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
                # Scan eerste 12 pagina's (Balans + W&V zitten altijd voorin)
                text = ""
                for i, page in enumerate(reader.pages):
                    if i < 12: text += page.extract_text()
                content += f"--- BRON: {pdf} ---\n{text[:6000]}\n\n"
                found = True
            except: pass
            
    if not found and intent['source'] == 'PDF':
        return "--- LET OP: Geen jaarrekeningen gevonden op GitHub. ---"
    return content

# ==========================================
# 3. INTELLIGENTIE LAAG (PROMPTS)
# ==========================================

def run_finny_mini(client, question, syllabus_extract):
    """
    IMPLEMENTATIE VAN PROMPT 9.6 / 9.7
    Vertaalt vraag naar RGS codes en zoektermen.
    """
    system_prompt = f"""
    Je bent Finny-Mini (Versie 9.6). Je bent een Query-Translator.
    Je geeft GEEN antwoord op de vraag. Je vertaalt de vraag naar database-parameters.
    
    HUIDIGE DATUM: {datetime.now().strftime("%Y-%m-%d")}
    
    JOUW KENNIS (Syllabus/RGS):
    {syllabus_extract}
    
    OPDRACHT:
    1. Analyseer de intentie: Wil de gebruiker een exact bedrag/lijstje (CSV) of inzicht/balans (PDF)?
    2. Vertaal 'gebruikerstaal' naar 'boekhoudtaal' (RGS).
       Vb: "Mijn auto" -> Zoektermen: ["Brandstof", "Wegenbelasting", "Garage", "Lease", "4100"]
    3. Bepaal het jaartal. "Vorig jaar" = {datetime.now().year - 1}.
    
    OUTPUT (JSON):
    {{
        "source": "CSV" of "PDF" of "BOTH",
        "years": [int],
        "search_terms": [list of strings],
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
        return json.loads(response.choices[0].message.content)
    except:
        return {"source": "BOTH", "years": [], "search_terms": []}

def run_finny_main(client, question, context):
    """
    IMPLEMENTATIE VAN PROMPT 9.9 (De Persona)
    """
    system_prompt = f"""
    Je bent Finny, de financiële AI-partner voor het MKB. (Conform Prompt 9.9).
    
    DATA CONTEXT:
    {context}
    
    RICHTLIJNEN VOOR JE ANTWOORD:
    1. **Start met de Conclusie:** Geef direct antwoord op de vraag. (Vb: "De totale autokosten in 2024 waren
