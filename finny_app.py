import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
import re
from datetime import datetime

# ==========================================
# 1. CONFIGURATIE & SETUP
# ==========================================
st.set_page_config(page_title="Finny | Intelligent Finance", page_icon="💰", layout="wide")

# Veilige wachtwoord check
def check_password():
    def password_entered():
        if st.session_state["password"] == "demo2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Probeer logo te tonen als het bestaat
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal (v7.0 Stable)</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# ==========================================
# 2. DATA CLEANING & LOADING (DE HARDCORE FIX)
# ==========================================
@st.cache_data
def load_and_clean_data():
    """
    Laadt alle data in het geheugen en poetst Nederlandse getallen op.
    Dit voorkomt de 'ValueError' crashes.
    """
    data = {
        "csv": None,
        "pdf_text": "",
        "syllabus": "",
        "status": "Bezig..."
    }

    # --- A. CSV LADEN & SCHOONMAKEN ---
    csv_file = "133700 FinTransactionSearch all 5jr.csv"
    if os.path.exists(csv_file):
        try:
            # Lees CSV met puntkomma
            df = pd.read_csv(csv_file, sep=";", on_bad_lines='skip', low_memory=False)
            
            # Kolommen normaliseren (hoofdletters/spaties weg)
            df.columns = df.columns.str.strip().str.lower()
            
            # Zoek de kritieke kolommen
            col_amt = next((c for c in df.columns if 'bedrag' in c or 'value' in c or 'amount' in c), None)
            col_date = next((c for c in df.columns if 'datum' in c or 'date' in c), None)
            
            if col_amt:
                # FUNCTIE: Maak van '€ 1.000,00' -> 1000.00 (Float)
                def clean_money(val):
                    if pd.isna(val): return 0.0
                    s = str(val).replace('€', '').replace(' ', '')
                    s = s.replace('.', '') # Duizendtal weg
                    s = s.replace(',', '.') # Komma naar punt
                    try: return float(s)
                    except: return 0.0
                
                df['clean_amount'] = df[col_amt].apply(clean_money)
            
            if col_date:
                # Zet om naar echte datums (dag-maand-jaar)
                df['clean_date'] = pd.to_datetime(df[col_date], dayfirst=True, errors='coerce')
                df['year'] = df['clean_date'].dt.year

            # Maak één doorzoekbare tekstkolom (voor synoniemen)
            text_cols = [c for c in df.columns if c not in ['clean_amount', 'clean_date', 'year']]
            df['search_text'] = df[text_cols].astype(str).agg(' '.join, axis=1).str.lower()
            
            data["csv"] = df
        except Exception as e:
            print(f"CSV Error: {e}")

    # --- B. PDF LADEN (JAARREKENINGEN) ---
    pdfs = [
        "Van Hattem Advies B.V. - Jaarrekening 2024.pdf",
        "Van Hattem Advies B.V. - Jaarrekening 2023.pdf",
        "Van Hattem Advies B.V. - Jaarstukken 2022.pdf"
    ]
    for pdf in pdfs:
        if os.path.exists(pdf):
            try:
                reader = PdfReader(pdf)
                text = ""
                # Scan eerste 20 pagina's (Balans/W&V)
                for i, page in enumerate(reader.pages):
                    if i < 20: text += page.extract_text()
                data["pdf_text"] += f"--- BRON: {pdf} ---\n{text[:10000]}\n\n"
            except: pass

    # --- C. SYLLABUS & PROFIEL ---
    if os.path.exists("Finny_syllabus.txt"):
        try: 
            with open("Finny_syllabus.txt", "r", encoding="utf-8") as f:
                data["syllabus"] += f"{f.read()}\n"
        except: pass
    if os.path.exists("van_hattem_advies_profiel.txt"):
        try: 
            with open("van_hattem_advies_profiel.txt", "r", encoding="utf-8") as f:
                data["syllabus"] += f"{f.read()}\n"
        except: pass

    return data

# ==========================================
# 3. DE INTELLIGENTIE (ROUTER & ENGINE)
# ==========================================

def run_router(client, question, syllabus_context):
    """
    Stap 1: Bepaal de strategie.
    Kijkt naar de vraag en de syllabus om te bepalen WAT we moeten zoeken.
    """
    system_prompt = f"""
    Je bent de Router van Finny.
    Huidige datum: {datetime.now().strftime("%Y-%m-%d")}
    
    CONTEXT (Syllabus/RGS):
    {syllabus_context[:3000]}

    JOUW DOEL:
    Vertaal de gebruikersvraag naar een zoekopdracht.
    
    REGELS:
    1. Als de vraag gaat over 'Omzet', 'Winst', 'Solvabiliteit', 'Balans' -> Bron = PDF
    2. Als de vraag gaat over 'Kosten', 'Bedragen', 'Leveranciers', 'Details' -> Bron = CSV
    3. VERTAAL zoektermen: "Auto" -> zoektermen ["brandstof", "lease", "garage", "4100"] (kijk naar RGS).
    
    ANTWOORD IN JSON:
    {{
        "source": "CSV" of "PDF" of "BOTH",
        "years": [2023, 2024], (of null voor alles)
        "keywords": ["term1", "term2"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"source": "BOTH", "years": [], "keywords": []}

def execute_csv_search(df, intent):
    """
    Stap 2: De Python Engine.
    Voert het filter uit en BEREKENT het totaal. Geen AI-rekenfouten meer.
    """
    if df is None: return "Geen transactiedata beschikbaar."
    
    # 1. Filter op Jaar
    filtered_df = df.copy()
    years = intent.get('years')
    if years:
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
        # Fallback: als jaarfilter 0 resultaten geeft, negeer het (misschien verkeerd jaar gegokt)
        if len(filtered_df) == 0:
            filtered_df = df.copy()
    
    # 2. Filter op Keywords (Synoniemen)
    keywords = intent.get('keywords')
    if keywords:
        # Maak regex pattern (veilig)
        safe_keywords = [re.escape(k.lower()) for k in keywords]
        pattern = '|'.join(safe_keywords)
        filtered_df = filtered_df[filtered_df['search_text'].str.contains(pattern, na=False)]
    
    # 3. Aggregatie (De Wiskunde)
    count = len(filtered_df)
    if count == 0:
        return f"Geen transacties gevonden voor: {keywords} in jaren {years}."
    
    total_amount = filtered_df['clean_amount'].sum()
    
    # Maak samenvatting voor de AI
    if count > 50:
        # Te veel regels? Geef top 10 posten
        top_desc = filtered_df['omschrijving'].value_counts().head(10).to_string()
        return f"""
        --- CSV ANALYSE RESULTAAT ---
        Gevonden transacties: {count}
        TOTAAL BEDRAG: € {total_amount:,.2f}
        
        Meest voorkomende omschrijvingen:
        {top_desc}
        """
    else:
        # Weinig regels? Geef details
        details = filtered_df[['datum', 'omschrijving', 'bedrag', 'grootboek']].to_string(index=False)
        return f"""
        --- CSV ANALYSE RESULTAAT ---
        Gevonden transacties: {count}
        TOTAAL BEDRAG: € {total_amount:,.2f}
        
        Details:
        {details}
        """

# ==========================================
# 4. DE APPLICATIE
# ==========================================
if check_password():
    # API Key Check
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt in Streamlit Secrets.")
        st.stop()

    # Data laden
    data = load_and_clean_data()

    # Sidebar
    with st.sidebar:
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        st.markdown("### 🏢 Van Hattem Advies B.V.")
        if data["csv"] is not None:
            st.success(f"✅ Live Data ({len(data['csv'])} boekingen)")
        else:
            st.error("❌ Geen CSV data")
        
        if st.button("Reset"): st.rerun()

    # Main Chat
    st.title("👋 Goedemiddag, Michiel.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Finny denkt na..."):
                # STAP 1: ROUTER
                intent = run_router(client, prompt, data["syllabus"])
                
                # Debug info (laat zien dat hij slim is)
                st.caption(f"🧠 Strategie: Bron={intent.get('source')} | Zoekt={intent.get('keywords')}")

                # STAP 2: DATA VERZAMELEN
                context = data["syllabus"] + "\n"
                
                if intent.get("source") in ["PDF", "BOTH"]:
                    context += data["pdf_text"]
                
                if intent.get("source") in ["CSV", "BOTH"]:
                    csv_result = execute_csv_search(data["csv"], intent)
                    context += csv_result

                # STAP 3: ANTWOORD
                system_msg = """
                Je bent Finny, de financiële assistent.
                Gebruik de DATA CONTEXT om antwoord te geven.
                
                BELANGRIJK:
                - Als er staat 'TOTAAL BEDRAG: € ...', gebruik dat cijfer. Ga niet zelf optellen.
                - Wees zakelijk en gebruik tabellen.
                - Zeg erbij waar je de info vandaan hebt (PDF of Transacties).
                """
                
                # Geheugen toevoegen (laatste 2 berichten)
                history = st.session_state.messages[-4:]
                msgs = [{"role": "system", "content": system_msg + f"\nDATA CONTEXT:\n{context}"}]
                for h in history:
                     msgs.append({"role": h["role"], "content": h["content"]})
                msgs.append({"role": "user", "content": prompt})

                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=msgs,
                        temperature=0
                    )
                    reply = resp.choices[0].message.content
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Fout: {e}")
