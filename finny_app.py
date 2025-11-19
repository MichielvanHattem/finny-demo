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
        # Probeer logo te tonen, faal stil als het niet lukt
        if os.path.exists("finny_logo.jpg"):
            st.image("finny_logo.jpg", width=150)
        
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        if os.path.exists("finny_logo.jpg"):
            st.image("finny_logo.jpg", width=150)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# --- DATA INLADEN ---
@st.cache_data
def load_demo_data():
    context = ""
    
    # Hulpfunctie om veilig te lezen
    def read_safely(filename, label, is_csv=False, is_pdf=False, sep=";"):
        content = ""
        if os.path.exists(filename):
            try:
                if is_csv:
                    df = pd.read_csv(filename, sep=sep, on_bad_lines='skip', low_memory=False)
                    content = f"--- {label} ---\n{df.head(3000).to_string(index=False)}\n\n"
                elif is_pdf:
                    reader = PdfReader(filename)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    content = f"--- {label} ({filename}) ---\n{text[:15000]}\n\n"
                else: # TXT
                    with open(filename, "r", encoding="utf-8") as f:
                        content = f"--- {label} ---\n{f.read()}\n\n"
            except: pass
        return content

    # Bestandsnamen EXACT zoals op GitHub
    context += read_safely("van_hattem_advies_profiel.txt", "KLANTPROFIEL")
    context += read_safely("Finny_syllabus.txt", "SYLLABUS")
    context += read_safely("133700 FinTransactionSearch all 5jr.csv", "TRANSACTIES", is_csv=True, sep=";")
    context += read_safely("133700 Standaard Rekeningschema Template FinGLAccountSearch.csv", "REKENINGSCHEMA", is_csv=True, sep="\t")
    
    pdfs = [
        "Van Hattem Advies B.V. - Jaarrekening 2024.pdf",
        "Van Hattem Advies B.V. - Jaarrekening 2023.pdf",
        "Van Hattem Advies B.V. - Jaarstukken 2022.pdf"
    ]
    for pdf in pdfs:
        context += read_safely(pdf, "JAARREKENING", is_pdf=True)

    return context

if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt")
        st.stop()

    with st.sidebar:
        if os.path.exists("finny_logo.jpg"):
            st.image("finny_logo.jpg", width=150)
        
        st.markdown("### 🏢 Van Hattem Advies B.V.")
        
        # --- DEBUG CHECK ---
        # Dit laat zien welke bestanden Streamlit écht ziet
        files_present = os.listdir()
        if "133700 FinTransactionSearch all 5jr.csv" in files_present:
             st.success("✅ Data Connectie Live")
        else:
             st.error("❌ Data nog aan het laden...")
             # Optioneel: st.write(files_present) # Zet dit aan als je wilt zien wat hij wél ziet
        
        st.markdown("---")
        if st.button("Reset Sessie"):
            st.rerun()

    st.title("👋 Goedemiddag, Michiel.")
    full_context = load_demo_data()
    
    if len(full_context) < 100:
        st.info("⚠️ Finny is aan het opstarten. Als dit blijft staan: Doe een 'Clear Cache'.")

    st.markdown("De administratie is bijgewerkt. Ik heb inzicht in je cijfers van 2022 t/m nu.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Finny denkt na..."):
                try:
                    messages = [
                        {"role": "system", "content": f"Je bent Finny. Antwoord zakelijk op basis van deze data:\n{full_context}"},
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
