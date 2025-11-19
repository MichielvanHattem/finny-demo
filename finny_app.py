import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
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
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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
        if st.button("Reset"):
            st.rerun()

    # --- HOOFDSCHERM ---
    st.markdown("### 👋 Hallo, ik ben Finny.")
    st.markdown("Ik ben jouw financiële assistent. Ik analyseer je jaarrekening en transacties direct.")
    
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
        context_text = ""

        # 1. Verwerk CSV (Simpel & Stabiel)
        if csv_file:
            try:
                csv_file.seek(0)
                try:
                    df = pd.read_csv(csv_file)
                except:
                    csv_file.seek(0)
                    df = pd.read_csv(csv_file, sep=';')
                
                # Zet de CSV om naar tekst zodat GPT het kan 'lezen'
                # We beperken tot de eerste 2000 regels voor snelheid en stabiliteit in de demo
                csv_string = df.head(2000).to_string(index=False)
                context_text += f"\n\nHIER ZIJN DE TRANSACTIES (CSV DATA):\n{csv_string}\n"
            except Exception as e:
                st.error(f"Kon CSV niet lezen: {e}")

        # 2. Verwerk PDF
        if pdf_file:
            try:
                pdf_text = ""
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                # Pak de eerste 30 pagina's (meestal genoeg voor jaarrekening)
                context_text += f"\n\nHIER IS DE JAARREKENING (PDF DATA):\n{pdf_text[:40000]}\n"
            except Exception as e:
                st.error(f"Kon PDF niet lezen: {e}")

        # 3. Stuur naar GPT-4o-mini
        if context_text == "":
            response_text = "Upload eerst een bestand zodat ik data heb om mee te werken."
        else:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Finny denkt na..."):
                    try:
                        messages = [
                            {"role": "system", "content": "Je bent Finny, een expert accountant. Je krijgt ruwe data (CSV transacties en/of PDF jaarrekening). \n\nJouw taak:\n1. Beantwoord de vraag van de gebruiker op basis van DEZE data.\n2. Als er transactiedata is, REKEN ZELF de totalen uit in je hoofd voordat je antwoordt.\n3. Geef korte, zakelijke antwoorden.\n4. Gebruik markdown tabellen als dat duidelijk is."},
                            {"role": "user", "content": f"Hier is de data:\n{context_text}\n\nDe vraag is: {prompt}"}
                        ]
                        
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini", # Gebruikt jouw beschikbare model
                            messages=messages,
                            temperature=0
                        )
                        response_text = completion.choices[0].message.content
                    except Exception as e:
                        response_text = f"Er ging iets mis bij de AI: {e}"

        st.chat_message("assistant").write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
