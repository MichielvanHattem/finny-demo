import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from PyPDF2 import PdfReader
import os

# --- CONFIGURATIE & HUISSTIJL ---
st.set_page_config(page_title="Finny | AI Financial Assistant", page_icon="💼", layout="wide")

# Verberg standaard Streamlit elementen voor een 'SaaS' look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .css-1rs6os {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- INLOG SYSTEEM (Fake authenticatie voor Demo Gevoel) ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "demo2025": # HET WACHTWOORD
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # wachtwoord niet bewaren
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Eerste keer, toon input
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Login</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.warning("🔒 Dit is een beveiligde omgeving voor financiële analyse.")
        return False
    elif not st.session_state["password_correct"]:
        # Fout wachtwoord
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("😕 Wachtwoord onjuist")
        return False
    else:
        # Goed wachtwoord
        return True

if check_password():
    # --- HIER BEGINT DE ECHTE APP NA INLOGGEN ---
    
    # Haal API key uit de cloud secrets (veilig)
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("Systeemfout: API Key ontbreekt in configuratie.")
        st.stop()

    # Sidebar (De 'Controller')
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=80) # Placeholder logo
        st.markdown("### 📂 Dossier Selectie")
        st.info("Selecteer de klantdata voor analyse.")
        
        pdf_file = st.file_uploader("Jaarrekening (PDF)", type=["pdf"])
        csv_file = st.file_uploader("Transacties (CSV)", type=["csv"])
        
        st.markdown("---")
        if st.button("Uitloggen"):
            st.session_state["password_correct"] = False
            st.rerun()

    # Hoofdscherm
    st.markdown("# 👋 Hallo, ik ben Finny.")
    st.markdown("##### *Jouw virtuele financieel analist.*")
    st.markdown("---")

    # Chat Geschiedenis
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ik heb toegang tot de geüploade dossiers. Stel me een vraag over de winstgevendheid, specifieke kostenposten of balanstotalen."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input Veld
    prompt = st.chat_input("Wat wil je weten over dit dossier?")

    if prompt:
        # Toon user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # --- HET BREIN (ROUTER) ---
        response_text = ""
        
        csv_triggers = ["hoeveel", "bedrag", "totaal", "som", "kosten", "omzet", "grootste", "leverancier", "transactie", "csv", "betaald", "rekening"]
        is_math = any(t in prompt.lower() for t in csv_triggers)

        # SCENARIO 1: REKENEN (CSV)
        if is_math and csv_file:
            with st.chat_message("assistant"):
                with st.status("🔍 Finny analyseert transacties...", expanded=True) as status:
                    try:
                        # CSV Inladen
                        csv_file.seek(0)
                        try:
                            df = pd.read_csv(csv_file)
                            if len(df.columns) < 2: df = pd.read_csv(csv_file, sep=';')
                        except:
                            df = pd.read_csv(csv_file, sep=';')
                        
                        # Agent Maken
                        llm = ChatOpenAI(temperature=0, model="gpt-4o")
                        agent = create_pandas_dataframe_agent(
                            llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, allow_dangerous_code=True
                        )
                        
                        status.write("Berekening uitvoeren...")
                        response = agent.run(prompt)
                        status.update(label="Berekening voltooid", state="complete", expanded=False)
                        response_text = response
                    except Exception as e:
                        status.update(label="Fout in berekening", state="error")
                        response_text = "Ik kon de berekening niet uitvoeren op deze dataset. Controleer het CSV formaat."

        # SCENARIO 2: LEZEN (PDF)
        elif pdf_file:
            with st.chat_message("assistant"):
                with st.spinner("📖 Finny leest de jaarrekening..."):
                    # PDF Tekst extraheren
                    text = ""
                    pdf_reader = PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    
                    # LLM Vraag
                    llm = ChatOpenAI(temperature=0.2, model="gpt-4o")
                    messages = [
                        {"role": "system", "content": f"Je bent Finny. Antwoord zakelijk en kort op basis van deze jaarrekening:\n\n{text[:40000]}"},
                        {"role": "user", "content": prompt}
                    ]
                    ai_msg = llm.invoke(messages)
                    response_text = ai_msg.content
        
        else:
            response_text = "Upload eerst een dossier (PDF of CSV) in het menu links om te kunnen starten."

        # Toon antwoord
        if not response_text: response_text = "Ik begreep de vraag niet goed in combinatie met de bestanden."
        
        # Check of we al geschreven hebben in de 'with' blocks
        if is_math and csv_file:
            st.write(response_text)
        elif pdf_file and not (is_math and csv_file): # PDF scenario
             st.write(response_text)
        elif not pdf_file and not csv_file:
             with st.chat_message("assistant"):
                st.write(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
