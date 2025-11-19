import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from PyPDF2 import PdfReader
import os

# --- CONFIGURATIE ---
st.set_page_config(page_title="Finny | AI Financial Assistant", page_icon="💼", layout="wide")

# --- INLOG SYSTEEM ---
def check_password():
    """Simpele wachtwoord beveiliging"""
    def password_entered():
        if st.session_state["password"] == "demo2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Login</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("😕 Wachtwoord onjuist")
        return False
    else:
        return True

if check_password():
    # API Key ophalen
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("Systeemfout: API Key ontbreekt in secrets.")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🤖 Finny")
        st.markdown("### Dossier Selectie")
        pdf_file = st.file_uploader("Jaarrekening (PDF)", type=["pdf"])
        csv_file = st.file_uploader("Transacties (CSV)", type=["csv"])
        
        st.markdown("---")
        if st.button("Herstarten"):
            st.rerun()

    # --- HOOFDSCHERM ---
    st.markdown("### 👋 Hallo, ik ben Finny.")
    st.markdown("Stel me vragen over de cijfers. Ik combineer boekhouding (PDF) met transacties (CSV).")
    
    # Chat historie
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload je bestanden en vraag maar raak!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input
    prompt = st.chat_input("Wat wil je weten?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        response_text = ""

        # --- LOGICA ---
        
        # CSV Vraag (PandasAI met gpt-4o-mini)
        if csv_file and any(x in prompt.lower() for x in ["hoeveel", "bedrag", "totaal", "som", "kosten", "omzet", "grootste", "transactie"]):
            with st.chat_message("assistant"):
                with st.spinner("🔍 Finny duikt in de transacties..."):
                    try:
                        csv_file.seek(0)
                        try:
                            df = pd.read_csv(csv_file)
                        except:
                            csv_file.seek(0)
                            df = pd.read_csv(csv_file, sep=';')
                        
                        # HIER IS DE WIJZIGING NAAR GPT-4o-MINI
                        llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
                        sdf = SmartDataframe(df, config={"llm": llm})
                        
                        response = sdf.chat(prompt)
                        response_text = str(response)
                        
                    except Exception as e:
                        response_text = f"Fout bij lezen CSV: {e}"

        # PDF Vraag (OpenAI call met gpt-4o-mini)
        elif pdf_file:
            with st.chat_message("assistant"):
                with st.spinner("📄 Finny leest de jaarrekening..."):
                    text = ""
                    pdf_reader = PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    
                    from openai import OpenAI as OpenAIClient
                    client = OpenAIClient()
                    
                    msg_history = [
                        {"role": "system", "content": f"Je bent een accountant. Antwoord op basis van deze jaarrekening:\n\n{text[:30000]}"},
                        {"role": "user", "content": prompt}
                    ]
                    # HIER IS DE WIJZIGING NAAR GPT-4o-MINI
                    completion = client.chat.completions.create(model="gpt-4o-mini", messages=msg_history)
                    response_text = completion.choices[0].message.content

        else:
            response_text = "Zorg dat je een bestand hebt geüpload (PDF of CSV)."

        st.chat_message("assistant").write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
