import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json

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
        # Logo check
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

# --- DATA FUNCTIES ---
# We laden data nu PAS als we het nodig hebben (Lazy Loading) om geheugen en tokens te sparen

def get_csv_content():
    """Haalt alleen de transactiedata op"""
    content = ""
    try:
        if os.path.exists("133700 FinTransactionSearch all 5jr.csv"):
            df = pd.read_csv("133700 FinTransactionSearch all 5jr.csv", sep=";", on_bad_lines='skip', low_memory=False)
            # We pakken 2000 regels, dat past makkelijk als we GEEN PDF's meesturen
            content += f"--- TRANSACTIES ---\n{df.head(2000).to_string(index=False)}\n\n"
        
        if os.path.exists("133700 Standaard Rekeningschema Template FinGLAccountSearch.csv"):
            df_ledger = pd.read_csv("133700 Standaard Rekeningschema Template FinGLAccountSearch.csv", sep="\t", on_bad_lines='skip')
            content += f"--- REKENINGSCHEMA ---\n{df_ledger.to_string(index=False)}\n\n"
    except: pass
    return content

def get_pdf_content():
    """Haalt alleen de jaarrekeningen op"""
    content = ""
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
                for page in reader.pages:
                    text += page.extract_text()
                content += f"--- {pdf} ---\n{text[:10000]}\n\n"
            except: pass
    return content

def get_base_content():
    """Haalt profiel en syllabus op (altijd nodig)"""
    content = ""
    try:
        with open("van_hattem_advies_profiel.txt", "r", encoding="utf-8") as f:
            content += f"--- KLANTPROFIEL ---\n{f.read()}\n\n"
        with open("Finny_syllabus.txt", "r", encoding="utf-8") as f:
            context += f"--- SYLLABUS ---\n{f.read()}\n\n"
    except: pass
    return content

# --- DE ROUTER (Jouw Finny-Mini idee) ---
def determine_intent(client, question):
    """Bepaalt of we CSV, PDF of niks nodig hebben"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                 Jij bent een router. Bepaal welke bron nodig is voor de vraag.
                 Antwoord ALLEEN met: 'CSV', 'PDF', of 'ALGEMEEN'.
                 - Vragen over transacties, bedragen, leveranciers, kosten, omzet details -> CSV
                 - Vragen over balans, winst-en-verlies, toelichting, solvabiliteit, kengetallen jaarrekening -> PDF
                 - Vragen over trends over jaren heen -> PDF
                 - Begroetingen of algemene vragen -> ALGEMEEN
                 """},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except:
        return "ALGEMEEN" # Fallback

if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt")
        st.stop()

    with st.sidebar:
        # Logo
        for ext in ["jpg", "jpeg", "png"]:
            if os.path.exists(f"finny_logo.{ext}"):
                st.image(f"finny_logo.{ext}", width=150)
                break
        
        st.markdown("### 🏢 Van Hattem Advies B.V.")
        
        # Status Check
        if os.path.exists("133700 FinTransactionSearch all 5jr.csv"):
             st.success("✅ Live Koppeling")
        else:
             st.error("❌ Data niet gevonden")
        
        st.markdown("---")
        if st.button("Reset Sessie"):
            st.rerun()

    st.title("👋 Goedemiddag, Michiel.")
    st.markdown("Ik ben ingelogd op Twinfield en je jaarrekeningendossier.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Finny bepaalt bron..."):
                
                # STAP 1: ROUTER (Kiezen welke data nodig is)
                intent = determine_intent(client, prompt)
                
                # STAP 2: DATA LADEN OP MAAT
                active_context = get_base_content() # Altijd profiel laden
                debug_msg = ""

                if "CSV" in intent:
                    active_context += get_csv_content()
                    debug_msg = "🔍 *Ik zoek in de transacties...*"
                elif "PDF" in intent:
                    active_context += get_pdf_content()
                    debug_msg = "📄 *Ik lees de jaarrekeningen...*"
                else:
                    debug_msg = "💭 *Algemene vraag...*"
                
                st.caption(debug_msg) # Laat gebruiker zien wat Finny doet (SaaS ervaring)

            with st.spinner("Analyseren..."):
                try:
                    messages = [
                        {"role": "system", "content": f"""
                        Je bent Finny.
                        Gebruik deze data om antwoord te geven:
                        {active_context}
                        
                        Stijl: Zakelijk, direct.
                        Als je transactiedata hebt: REKEN ZELF totalen uit.
                        Als je PDF data hebt: Citeer of gebruik de cijfers uit de tabel.
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
                    if "429" in str(e):
                        st.error("⚠️ Demo Limiet: Finny is druk. Wacht even of upgrade je OpenAI account.")
                    else:
                        st.error(f"Fout: {e}")
