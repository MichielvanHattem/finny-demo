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
st.set_page_config(page_title="Finny 5.3 | Lite & Fast", page_icon="💰", layout="wide")

def check_password():
    """Eenvoudige wachtwoordbeveiliging."""
    def password_entered():
        if st.session_state["password"] == "demo2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        if os.path.exists("finny_logo.jpg"):
            st.image("finny_logo.jpg", width=150)
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# ==========================================
# 2. DATA LADEN (NIEUWE LITE BESTANDEN)
# ==========================================
@st.cache_data
def load_data():
    data = {"trans": None, "ledger": None, "syllabus": "", "pdf_text": ""}
    
    # 1. Transacties (De nieuwe Lite CSV)
    # We gebruiken nu de schone kolommen: Finny_GLCode, AmountDC_num, Finny_Year
    if os.path.exists("Finny_Transactions_Lite.csv"):
        try:
            # Let op de separator, vaak ; bij Nederlandse CSV's, check je bestand!
            # Op basis van je upload lijkt het ; te zijn.
            df = pd.read_csv("Finny_Transactions_Lite.csv", sep=";", on_bad_lines='skip', low_memory=False)
            # Zorg dat datum een datetime object is
            if 'EntryDate' in df.columns:
                df['EntryDate'] = pd.to_datetime(df['EntryDate'], errors='coerce')
            data["trans"] = df
        except Exception as e:
            st.error(f"Fout bij laden transacties: {e}")

    # 2. Grootboekschema (Lite CSV)
    if os.path.exists("Finny_GL_Lite.csv"):
        try:
            df_l = pd.read_csv("Finny_GL_Lite.csv", sep=";", on_bad_lines='skip')
            data["ledger"] = df_l
        except Exception as e:
            st.error(f"Fout bij laden grootboek: {e}")

    # 3. Syllabus (Voor de synoniemen)
    if os.path.exists("Finny_syllabus_v9_4_with_B3_B4_B5.txt"):
        try:
            with open("Finny_syllabus_v9_4_with_B3_B4_B5.txt", "r", encoding="utf-8") as f:
                data["syllabus"] = f.read()
        except:
            pass
            
    # 4. Jaarrekeningen (PDF)
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            # Eerste 15 pagina's zijn vaak de kerncijfers
            for i, page in enumerate(reader.pages):
                if i < 15: text += page.extract_text()
            data["pdf_text"] += f"--- BRON: {pdf} ---\n{text[:8000]}\n\n"
        except:
            pass
            
    return data

# ==========================================
# 3. DE LOGICA (ROUTER & QUERY)
# ==========================================

def get_finny_plan(client, question, syllabus):
    """
    Stap 1 van je proces: Vraag -> Synoniemen -> Grootboekrekening(en).
    """
    system_prompt = f"""
    Je bent het brein van Finny. Je analyseert de vraag.
    CONTEXT (Syllabus met synoniemen):
    {syllabus[:5000]} 
    
    TAAK:
    1. Bepaal of de vraag over 'PDF' (Jaarrekening/Winst/Balans totaal) of 'CSV' (Specifieke kosten/leveranciers/details) gaat.
    2. Als CSV: Welke 'Finny_GLCode's' (grootboeknummers) of zoektermen zijn relevant?
    3. Welke jaren? (Huidig jaar is {datetime.now().year}).
    
    OUTPUT (JSON):
    {{
        "source": "PDF" of "CSV" of "BOTH",
        "gl_codes": [lijst van ints, bijv 4300],
        "search_terms": [lijst van strings, bijv "Vodafone"],
        "years": [lijst van ints],
        "reason": "Korte uitleg"
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
        return {"source": "BOTH", "gl_codes": [], "search_terms": [], "years": []}

def execute_csv_query(plan, df):
    """
    Stap 2 & 3: Filteren en Aggregeren.
    Gebruikt de nieuwe 'Finny_Transactions_Lite.csv' structuur.
    """
    if df is None: return ""
    
    subset = df.copy()
    
    # 1. Filter op Jaar (Finny_Year)
    if plan.get("years"):
        subset = subset[subset['Finny_Year'].isin(plan['years'])]
        
    # 2. Filter op GL Codes (Finny_GLCode)
    # We moeten zorgen dat de types matchen (float/int/str probleem voorkomen)
    if plan.get("gl_codes"):
        # Maak schoon en converteer naar string voor matching
        codes = [str(c).split('.')[0] for c in plan['gl_codes']] 
        subset['GL_Str'] = subset['Finny_GLCode'].astype(str).str.split('.').str[0]
        subset = subset[subset['GL_Str'].isin(codes)]
        
    # 3. Filter op Zoektermen (Description / AccountName)
    if plan.get("search_terms"):
        terms = plan['search_terms']
        pattern = '|'.join(terms)
        # Zoek in Omschrijving én Relatienaam
        mask = (
            subset['Description'].astype(str).str.contains(pattern, case=False, na=False) | 
            subset['AccountName'].astype(str).str.contains(pattern, case=False, na=False) |
            subset['Finny_GLDescription'].astype(str).str.contains(pattern, case=False, na=False)
        )
        # Als er ook GL codes waren, is dit een EN (verfijning), anders een OF
        if not plan.get("gl_codes"):
             subset = subset[mask]
        # Als we al GL codes hadden, filteren we verder binnen die selectie als de gebruiker dat specifiek vraagt, 
        # maar meestal zijn GL codes leidend. Laten we zoektermen gebruiken als GL codes leeg zijn of als verfijning.

    # 4. Resultaat (Aggregatie)
    if len(subset) == 0:
        return "Geen transacties gevonden."
        
    # Totaal berekenen (AmountDC_num is al schoon!)
    total = subset['AmountDC_num'].sum()
    
    # Samenvatting maken
    if len(subset) > 50:
        # Groepeer per GL Omschrijving
        summary = subset.groupby('Finny_GLDescription')['AmountDC_num'].sum().reset_index()
        return f"""
        --- CSV ANALYSE ---
        Gevonden boekingen: {len(subset)}
        TOTAAL BEDRAG: € {total:,.2f}
        
        Uitsplitsing per categorie:
        {summary.to_string(index=False)}
        """
    else:
        # Details tonen
        cols = ['EntryDate', 'AccountName', 'Description', 'AmountDC_num']
        return f"""
        --- CSV DETAILS ---
        TOTAAL: € {total:,.2f}
        
        {subset[cols].to_string(index=False)}
        """

# ==========================================
# 4. MAIN APP
# ==========================================
if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt in secrets.")
        st.stop()

    # Sidebar
    with st.sidebar:
        if os.path.exists("finny_logo.jpg"):
            st.image("finny_logo.jpg", width=150)
        st.markdown("### 🏢 Van Hattem Advies")
        
        # Laad data
        data = load_data()
        
        if data["trans"] is not None:
            st.success(f" ✅ Transacties ({len(data['trans'])} regels)")
        else:
            st.error(" ❌ CSV niet gevonden")
        
        if st.button("Reset Sessie"): st.rerun()

    # Chat
    st.title(" 👋 Goedemiddag, Michiel.")
    st.markdown("Ik heb inzicht in je cijfers t/m vandaag.")

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Finny denkt na..."):
                # Stap 1: Plan
                plan = get_finny_plan(client, prompt, data["syllabus"])
                
                # Debug info voor jou (zodat je ziet dat hij de syllabus gebruikt)
                st.caption(f" 🧠 Strategie: {plan['source']} | Zoekt: {plan['gl_codes']} / {plan['search_terms']}")
                
                # Stap 2: Data verzamelen
                context = ""
                if plan["source"] in ["CSV", "BOTH"]:
                    context += execute_csv_query(plan, data["trans"])
                if plan["source"] in ["PDF", "BOTH"]:
                    context += data["pdf_text"]
                
                # Stap 3: Antwoord
                try:
                    messages = [
                        {"role": "system", "content": f"""
                        Je bent Finny. Antwoord op basis van deze data:
                        {context}
                        
                        REGELS:
                        1. Als er CSV data is met een totaalbedrag, gebruik dat EXACT. Reken niet zelf opnieuw.
                        2. Als de vraag over de jaarrekening gaat, citeer de PDF.
                        3. Wees zakelijk en kort.
                        """},
                        {"role": "user", "content": prompt}
                    ]
                    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
                    response = completion.choices[0].message.content
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Fout: {e}")
