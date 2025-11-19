import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os

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
        # LOGO op inlogpagina (indien aanwezig)
        try:
            st.image("finny_logo.png", width=150)
        except:
            pass # Geen logo? Geen probleem.
            
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        try:
            st.image("finny_logo.png", width=150)
        except:
            pass
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# --- DATA INLADEN (ACHTER DE SCHERMEN) ---
@st.cache_data
def load_demo_data():
    context = ""
    
    # 1. KLANTPROFIEL & SYLLABUS
    try:
        with open("van_hattem_advies_profiel.txt", "r", encoding="utf-8") as f:
            context += f"--- KLANTPROFIEL ---\n{f.read()}\n\n"
    except: pass

    try:
        with open("Finny_syllabus.txt", "r", encoding="utf-8") as f:
            context += f"--- FINNY SYLLABUS (OPERATIELE REGELS) ---\n{f.read()}\n\n"
    except: pass

    # 2. TRANSACTIES (CSV)
    try:
        # Lees de 5-jaars historie
        df_trans = pd.read_csv("133700 FinTransactionSearch all 5jr.csv", sep=";", on_bad_lines='skip', low_memory=False)
        # We pakken iets meer regels voor betere historie, let op token limiet
        csv_text = df_trans.head(4000).to_string(index=False) 
        context += f"--- TRANSACTIE DATA (CSV) ---\n{csv_text}\n\n"
        
        df_ledger = pd.read_csv("133700 Standaard Rekeningschema Template FinGLAccountSearch.csv", sep="\t", on_bad_lines='skip')
        ledger_text = df_ledger.to_string(index=False)
        context += f"--- REKENINGSCHEMA ---\n{ledger_text}\n\n"
    except Exception as e:
        st.error(f"Fout bij laden CSV: {e}")

    # 3. JAARREKENINGEN (PDF) - NU ALLE DRIE DE JAREN
    pdf_files = [
        "Van Hattem Advies B.V. - Jaarrekening 2024.pdf",
        "Van Hattem Advies B.V. - Jaarrekening 2023.pdf",
        "Van Hattem Advies B.V. - Jaarstukken 2022.pdf"  # Toegevoegd!
    ]
    
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            # Voeg jaar toe aan context label
            context += f"--- INHOUD BESTAND: {pdf_file} ---\n{text[:15000]}\n\n" 
        except Exception as e:
            st.warning(f"Kon {pdf_file} niet vinden op GitHub.")
            
    return context

if check_password():
    # API Setup
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt in Streamlit Secrets.")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        try:
            st.image("finny_logo.png", width=150)
        except:
            st.markdown("### 🤖 Finny")
            
        st.markdown("### 🏢 Van Hattem Advies B.V.")
        st.success("✅ Live Twinfield Koppeling")
        st.success("✅ Jaarrekeningen '22, '23, '24")
        st.success("✅ Transacties (5 jaar)")
        
        st.markdown("---")
        if st.button("Sessie Resetten"):
            st.rerun()

    # --- HOOFDSCHERM ---
    st.title("👋 Goedemiddag, Michiel.")
    st.markdown("De administratie is bijgewerkt. Ik heb inzicht in je cijfers van 2022 t/m nu.")

    full_context = load_demo_data()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Bijv: Hoe ontwikkelt mijn omzet zich over de laatste 3 jaar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Finny analyseert 3 jaar aan data..."):
                try:
                    # System Prompt geoptimaliseerd voor trendanalyse
                    messages = [
                        {"role": "system", "content": f"""
                        Je bent Finny, dé financiële AI-partner.
                        
                        JOUW KRACHT:
                        Je hebt toegang tot de volledige administratie (2022-2024) en transacties.
                        Gebruik deze data om trends te signaleren. Als iemand vraagt naar "ontwikkeling" of "vergelijking", zet dan de jaren naast elkaar.
                        
                        DATA CONTEXT:
                        {full_context}
                        
                        ANTWOORD STIJL:
                        - Zakelijk, direct, proactief.
                        - Reken zelf totalen uit op basis van de CSV als dat nodig is.
                        - Gebruik Markdown tabellen voor cijferoverzichten.
                        - Gebruik punten voor duizendtallen (€ 1.000).
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
