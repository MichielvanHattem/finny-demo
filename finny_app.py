import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
import datetime

# --- CONFIGURATIE ---
st.set_page_config(page_title="Finny | Intelligent Finance", page_icon="💰", layout="wide")

# --- INLOG SYSTEEM ---
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
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# --- DATA LADEN (HELE DATASET IN GEHEUGEN) ---
@st.cache_data
def load_full_data():
    data = {"trans": None, "ledger": None, "pdfs": ""}
    
    # 1. Transacties (Twinfield)
    if os.path.exists("133700 FinTransactionSearch all 5jr.csv"):
        try:
            # We lezen ALLES in, Pandas kan miljoenen regels aan, dat is het probleem niet.
            # Het probleem is tokens naar OpenAI sturen.
            df = pd.read_csv("133700 FinTransactionSearch all 5jr.csv", sep=";", on_bad_lines='skip', low_memory=False)
            
            # Kolommen normaliseren (Twinfield heeft rare namen)
            # We zoeken de kolom die 'datum' heet en maken die echt datum
            for col in df.columns:
                if 'datum' in col.lower() or 'date' in col.lower():
                    df['Datum_Clean'] = pd.to_datetime(df[col], errors='coerce')
                if 'bedrag' in col.lower() or 'value' in col.lower():
                    df['Bedrag_Clean'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                if 'omschrijving' in col.lower() or 'desc' in col.lower():
                    df['Omschrijving_Clean'] = df[col].astype(str)
                if 'grootboek' in col.lower() or 'dim1' in col.lower(): # Dim1 is vaak grootboek in Twinfield
                    df['Grootboek_Clean'] = df[col].astype(str)
            
            data["trans"] = df
        except Exception as e:
            st.error(f"CSV Laadfout: {e}")

    # 2. Rekeningschema
    if os.path.exists("133700 Standaard Rekeningschema Template FinGLAccountSearch.csv"):
        try:
            df_l = pd.read_csv("133700 Standaard Rekeningschema Template FinGLAccountSearch.csv", sep="\t", on_bad_lines='skip')
            data["ledger"] = df_l
        except: pass

    # 3. PDF's en Tekst
    pdfs = ["Van Hattem Advies B.V. - Jaarrekening 2024.pdf", "Van Hattem Advies B.V. - Jaarrekening 2023.pdf", "Van Hattem Advies B.V. - Jaarstukken 2022.pdf"]
    for pdf in pdfs:
        if os.path.exists(pdf):
            try:
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages: text += page.extract_text()
                data["pdfs"] += f"--- {pdf} (SAMENVATTING) ---\n{text[:8000]}\n\n"
            except: pass
    
    try:
        with open("van_hattem_advies_profiel.txt", "r", encoding="utf-8") as f:
            data["pdfs"] += f"--- PROFIEL ---\n{f.read()}\n\n"
        with open("Finny_syllabus.txt", "r", encoding="utf-8") as f:
            data["pdfs"] += f"--- SYLLABUS ---\n{f.read()}\n\n"
    except: pass

    return data

# --- DE SLIMME FILTER ENGINE ---
def smart_filter_transactions(client, question, df, df_ledger):
    """
    Dit is het brein.
    1. Vraagt LLM: 'Welke filters heb ik nodig?'
    2. Filtert de Pandas DataFrame
    3. Als het nog te groot is -> Aggregeert data (Totalen per maand/categorie)
    """
    
    # STAP 1: BEPAAL FILTER STRATEGIE
    ledger_summary = ""
    if df_ledger is not None:
        # We geven de eerste 50 regels van het rekeningschema mee als context voor zoektermen
        ledger_summary = df_ledger.head(50).to_string()

    system_prompt = f"""
    Jij bent een SQL-expert voor boekhouding.
    Vertaal de gebruikersvraag naar zoekparameters voor een Pandas dataframe.
    
    Output MOET geldige JSON zijn met deze velden:
    - "years": lijst met jaren (int) of null als alle jaren.
    - "keywords": lijst met zoektermen (strings) voor omschrijving/relatie.
    - "categories": lijst met mogelijke grootboekrekening-namen of codes (kijk naar context).
    
    Context (Rekeningschema):
    {ledger_summary}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        filters = json.loads(response.choices[0].message.content)
    except:
        filters = {"years": None, "keywords": [], "categories": []} # Fallback

    # STAP 2: PAS FILTERS TOE OP DATAFRAME
    filtered_df = df.copy()
    
    # Filter op Jaar
    if filters.get("years"):
        # Zorg dat we filteren op de datum kolom
        filtered_df = filtered_df[filtered_df['Datum_Clean'].dt.year.isin(filters['years'])]
    
    # Filter op Trefwoorden (Omschrijving OF Grootboek)
    search_terms = filters.get("keywords", []) + filters.get("categories", [])
    if search_terms:
        # We maken een regex die zoekt op term1 OR term2 OR term3
        pattern = '|'.join(search_terms)
        # Case insensitive search in Omschrijving EN Grootboek
        mask = (
            filtered_df['Omschrijving_Clean'].str.contains(pattern, case=False, na=False) | 
            filtered_df['Grootboek_Clean'].str.contains(pattern, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    # STAP 3: AGGREGEREN ALS DATA NOG STEEDS TE GROOT IS
    # Als we na filteren nog steeds > 200 regels hebben (bijv. "Totale omzet"),
    # Dan tellen we het alvast op in Python. Dit bespaart tokens én rekenfouten.
    row_count = len(filtered_df)
    
    if row_count > 200:
        # Groepeer per Jaar en Grootboekrekening
        try:
            # Voeg jaar kolom toe voor groepering
            filtered_df['Jaar'] = filtered_df['Datum_Clean'].dt.year
            grouped = filtered_df.groupby(['Jaar', 'Grootboek_Clean'])['Bedrag_Clean'].sum().reset_index()
            return f"--- GEAGGREGEERDE DATA (Oorspronkelijk {row_count} regels) ---\n{grouped.to_string(index=False)}\n\n", filters
        except:
            return f"--- DATA (Eerste 200 van {row_count}) ---\n{filtered_df.head(200).to_string(index=False)}\n\n", filters
    
    elif row_count == 0:
         return "--- GEEN TRANSACTIES GEVONDEN MET DIT FILTER ---\n\n", filters
    else:
        # Als het klein genoeg is, geven we de details
        cols_to_show = ['Datum_Clean', 'Omschrijving_Clean', 'Bedrag_Clean', 'Grootboek_Clean']
        # Pak alleen kolommen die bestaan
        final_cols = [c for c in cols_to_show if c in filtered_df.columns]
        return f"--- TRANSACTIE DETAILS ---\n{filtered_df[final_cols].to_string(index=False)}\n\n", filters


if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt")
        st.stop()

    # Sidebar
    with st.sidebar:
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("### 🏢 Van Hattem Advies B.V.")
        
        if os.path.exists("133700 FinTransactionSearch all 5jr.csv"):
             st.success("✅ Twinfield Live (5jr)")
        else:
             st.error("❌ Data Connectie Fout")
        
        st.markdown("---")
        if st.button("Reset"): st.rerun()

    st.title("👋 Goedemiddag, Michiel.")
    
    # LAAD DATA (1x)
    full_data = load_full_data()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            
            # 1. BEPAAL OF WE CSV OF PDF NODIG HEBBEN
            # Simpele router: Als vraag 'kosten', 'bedrag', 'leverancier', 'omzet' bevat -> CSV
            csv_keywords = ["hoeveel", "kosten", "bedrag", "betaald", "ontvangen", "omzet", "leverancier", "grootboek", "transactie"]
            is_csv_intent = any(k in prompt.lower() for k in csv_keywords)
            
            context_data = full_data["pdfs"] # Profiel en syllabus altijd meegeven
            
            if is_csv_intent and full_data["trans"] is not None:
                with st.spinner("Finny filtert de transacties..."):
                    # AANROEPEN SLIMME FILTER
                    csv_context, used_filters = smart_filter_transactions(client, prompt, full_data["trans"], full_data["ledger"])
                    context_data += csv_context
                    
                    # Toon de gebruiker wat we doen (SaaS magic)
                    filter_msg = "🔍 *"
                    if used_filters.get('years'): filter_msg += f"Jaar: {used_filters['years']} "
                    if used_filters.get('keywords'): filter_msg += f"Zoekterm: {used_filters['keywords']}"
                    filter_msg += "*"
                    st.caption(filter_msg)
            
            with st.spinner("Analyseren..."):
                try:
                    messages = [
                        {"role": "system", "content": f"""
                        Je bent Finny.
                        Gebruik onderstaande gefilterde data om de vraag te beantwoorden.
                        
                        DATA:
                        {context_data}
                        
                        INSTRUCTIES:
                        1. Als je een tabel met 'GEAGGREGEERDE DATA' ziet, zijn de optelsommen al voor je gemaakt. Gebruik die.
                        2. Als je 'TRANSACTIE DETAILS' ziet, moet jij de bedragen optellen.
                        3. Wees zakelijk, gebruik Markdown tabellen.
                        """},
                        {"role": "user", "content": prompt}
                    ]
                    
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0
                    )
                    response = completion.choices[0].message.content
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Fout: {e}")
