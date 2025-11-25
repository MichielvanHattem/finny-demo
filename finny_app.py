import streamlit as st
import pandas as pd
import re
from datetime import datetime

# --- CONFIGURATIE ---
st.set_page_config(page_title="Finny Demo", layout="wide")

# --- 1. INITIALISATIE & STATE MANAGEMENT ---

def init_state():
    """Initialiseer alle sessie-variabelen."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hallo! Ik ben Finny. Ik help je met inzichten uit je jaarrekening (PDF) en transacties (CSV)."}
        ]
    
    if "fact_cache" not in st.session_state:
        st.session_state.fact_cache = {}
    
    if "active_years" not in st.session_state:
        st.session_state.active_years = [datetime.now().year - 1] 
    
    if "client_profile" not in st.session_state:
        st.session_state.client_profile = {}

init_state()

# --- 2. DATA LADEN (Strikt & Veilig) ---

@st.cache_data
def load_data():
    """
    Laadt data uit CSV's.
    PRINCIPE: We passen data NIET inhoudelijk aan.
    Regels die technisch niet kunnen (bijv. geen jaartal) worden genegeerd (dropped),
    in plaats van opgevuld met fictieve waarden.
    """
    try:
        # 1. TRANSACTIES
        # We lezen alles eerst als string (dtype=str) om crashes te voorkomen bij het inlezen.
        # Zo blijft de originele data intact totdat we expliciet converteren.
        df_trans = pd.read_csv(
            "Finny_Transactions.csv", 
            sep=";", 
            encoding="latin-1",
            dtype=str 
        )
        
        # CONVERSIE STAP (Noodzakelijk voor berekeningen)
        
        # A. Jaartal: Moet een getal zijn om op te filteren.
        # 'coerce' maakt ongeldige waarden (tekst/leeg) NaN (Not a Number).
        df_trans['Finny_Year'] = pd.to_numeric(df_trans['Finny_Year'], errors='coerce')
        
        # B. Bedrag: Python heeft punten nodig voor decimalen (US format).
        # We checken eerst of het een string is, vervangen dan duizendtal-punten en decimaal-komma's.
        if df_trans['AmountDC_num'].dtype == object:
            df_trans['AmountDC_num'] = (
                df_trans['AmountDC_num']
                .str.replace('.', '', regex=False)  # Verwijder duizendtal punten
                .str.replace(',', '.', regex=False) # Vervang komma door punt
            )
        df_trans['AmountDC_num'] = pd.to_numeric(df_trans['AmountDC_num'], errors='coerce')

        # C. SCHONING: Verwijder regels die nu nog ongeldig zijn.
        # In plaats van 'fillna(0)' (data aanpassen), gooien we rommel weg.
        initial_len = len(df_trans)
        df_trans = df_trans.dropna(subset=['Finny_Year', 'AmountDC_num'])
        
        # Zet het jaar nu netjes om naar integer (kan veilig na dropna)
        df_trans['Finny_Year'] = df_trans['Finny_Year'].astype(int)

        # 2. SYNONIEMEN
        df_syn = pd.read_csv("Finny_Synonyms.csv", sep=";", encoding="latin-1", dtype=str)
        
        # 3. RGS CODES
        df_rgs = pd.read_csv("Finny_RGS.csv", sep=";", encoding="latin-1", dtype={'Finny_GLCode': str})
        
        return df_trans, df_syn, df_rgs

    except FileNotFoundError:
        st.warning("⚠️ Geen CSV-bestanden gevonden. Finny draait in DEMO-modus met testdata.")
        
        # Veilige mock data
        df_trans = pd.DataFrame({
            'Finny_Year': [2024, 2024, 2024, 2024, 2023],
            'Finny_GLCode': ['4001', '4001', '4050', '8000', '4001'],
            'AmountDC_num': [50.00, 2142.78, 150.00, 50000.00, 1200.00],
            'Description': ['Vodafone', 'KPN Zakelijk', 'Lunch', 'Omzet', 'Vodafone']
        })
        
        df_syn = pd.DataFrame({
            'Keyword': ['telefoon', 'gsm', 'vodafone', 'ziggo', 'kpn', 'mobiel'],
            'Category': ['telefoonkosten', 'telefoonkosten', 'telefoonkosten', 'telefoonkosten', 'telefoonkosten', 'telefoonkosten']
        })
        
        df_rgs = pd.DataFrame({
            'Category': ['telefoonkosten', 'telefoonkosten', 'omzet'],
            'Finny_GLCode': ['4001', '4002', '8000']
        })
        
        return df_trans, df_syn, df_rgs

df_trans, df_syn, df_rgs = load_data()

# --- 3. INTENT & ROUTER LOGICA ---

def extract_years(text):
    years = re.findall(r'\b(202[0-9])\b', text)
    if years:
        unique_years = sorted(list(set([int(y) for y in years])))
        st.session_state.active_years = unique_years
        return unique_years
    return st.session_state.active_years

def determine_intent(user_input):
    user_input_lower = user_input.lower()
    years = extract_years(user_input)
    
    csv_keywords = ["telefoon", "gsm", "vodafone", "ziggo", "kpn", "transacties", "boekingen", "facturen", "kosten per", "leverancier", "top 10"]
    pdf_keywords = ["winst", "omzet", "resultaat", "totaal kosten", "jaarrekening", "balans", "passiva"]
    
    intent = {
        "source": "PDF",
        "metric": "general",
        "category": None,
        "years": years
    }
    
    if any(kw in user_input_lower for kw in csv_keywords):
        intent["source"] = "CSV"
        intent["metric"] = "kosten"
        found_cat = None
        for idx, row in df_syn.iterrows():
            if str(row['Keyword']) in user_input_lower: # str() voor veiligheid
                found_cat = row['Category']
                break
        
        intent["category"] = found_cat if found_cat else "general_transactions"

    elif any(kw in user_input_lower for kw in pdf_keywords):
        intent["source"] = "PDF"
        if "winst" in user_input_lower: intent["metric"] = "winst"
        if "omzet" in user_input_lower: intent["metric"] = "omzet"
    
    return intent

# --- 4. CALCULATIE ENGINE (CSV) ---

def get_csv_metrics(intent):
    category = intent["category"]
    year = intent["years"][0]
    metric = intent["metric"]
    
    cache_key = (metric, category, year)
    if cache_key in st.session_state.fact_cache:
        cached_data = st.session_state.fact_cache[cache_key]
        return f"FACT (uit geheugen): {cached_data['text']}"

    if category == "general_transactions":
        return "CONTEXT: Ik zie dat je transacties wilt zien, maar ik kan de specifieke categorie niet bepalen. Vraag bijvoorbeeld specifiek naar 'telefoonkosten'."
    
    target_gl_codes = df_rgs[df_rgs['Category'] == category]['Finny_GLCode'].tolist()
    
    if not target_gl_codes:
        msg = f"In de administratie is geen specifieke grootboekrekening gekoppeld aan de categorie '{category}'. Ik kan daarom geen betrouwbaar totaalbedrag geven."
        st.session_state.fact_cache[cache_key] = {'amount': 0, 'source': 'CSV', 'text': msg}
        return f"FACT: {msg}"

    mask_year = df_trans['Finny_Year'] == year
    mask_gl = df_trans['Finny_GLCode'].astype(str).isin([str(c) for c in target_gl_codes])
    
    filtered_df = df_trans[mask_year & mask_gl]
    
    total_amount = filtered_df['AmountDC_num'].sum()
    count = len(filtered_df)
    
    if count > 0:
        fact_text = f"De totale {category} in {year} bedragen € {total_amount:,.2f}. Dit is de som van {count} boekingen op de grootboekrekeningen voor {category}."
    else:
        fact_text = f"Er zijn in {year} geen boekingen gevonden voor de categorie {category}."

    st.session_state.fact_cache[cache_key] = {
        'amount': total_amount,
        'source': 'CSV',
        'text': fact_text
    }
    
    return f"FACT (Nieuw berekend): {fact_text}"

# --- 5. PDF ENGINE (PLACEHOLDER) ---

def get_pdf_metrics(intent):
    """
    Haalt info uit de PDF/Vectorstore.
    PLAK HIER JE OUDE PDF FUNCTIE TERUG.
    """
    try:
        # MOCK VOOR DEMO
        year = intent["years"][0]
        if intent["metric"] == "winst":
            return f"FACT: In {year} was de winst na belastingen € 76.226 (volgens de jaarrekening PDF)."
        elif intent["metric"] == "omzet":
            return f"FACT: De omzet in {year} bedroeg € 120.000 (volgens de jaarrekening PDF)."
        else:
            return "CONTEXT: Ik heb gezocht in de PDF, maar vond geen exact antwoord op deze specifieke vraag."
            
    except Exception as e:
        return f"FOUT BIJ PDF ZOEKEN: {str(e)}"

# --- 6. LLM INTEGRATIE ---

def generate_llm_response(intent, context_text):
    system_prompt = f"""
    Je bent Finny, een financiële assistent.
    
    INSTRUCTIES:
    1. Baseer je antwoord UITSLUITEND op de onderstaande CONTEXT.
    2. Als de context begint met "FACT", is dit een hard, berekend cijfer. Neem dit bedrag EXACT over.
    3. Ga NOOIT zelf bedragen optellen of verzinnen.
    4. Als de context zegt "geen boekingen", zeg dan dat je het niet weet. Ga NIET gokken.
    
    CONTEXT:
    {context_text}
    """
    
    # MOCK RESPONSE (Vervang dit door jouw openai.ChatCompletion.create)
    if "FACT" in context_text:
        clean_fact = context_text.split("):")[-1].strip()
        if intent["source"] == "CSV":
            return f"{clean_fact}"
        else:
            return f"{clean_fact} (Gevonden in de jaarrekening)."
    else:
        return "Ik heb gezocht in de documenten, maar kon geen specifiek bedrag vinden voor je vraag."

# --- 7. UI VIEWS ---

def view_chat():
    st.header("Chat met Finny")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Stel je vraag (bijv. 'Wat zijn mijn telefoonkosten 2024?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        intent = determine_intent(prompt)
        
        if intent["source"] == "CSV":
            context = get_csv_metrics(intent)
        else:
            context = get_pdf_metrics(intent)
            
        response = generate_llm_response(intent, context)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
            
        with st.expander("🕵️ Debug Info (Intent & Cache)"):
            st.write(f"**Intent:** {intent}")
            st.write(f"**Context:** {context}")
            st.write("**Fact Cache Inhoud:**", st.session_state.fact_cache)

def view_intro():
    st.header("Kennismaking met Finny")
    with st.form("profile_form"):
        name = st.text_input("Je naam", value=st.session_state.client_profile.get("name", ""))
        role = st.selectbox("Je rol", ["DGA", "Boekhouder", "Investeerder"], index=0)
        focus = st.multiselect("Waar let je op?", ["Winst", "Kostenbeheersing", "Groei"], default=["Winst"])
        
        submitted = st.form_submit_button("Opslaan")
        if submitted:
            st.session_state.client_profile = {"name": name, "role": role, "focus": focus}
            st.success("Profiel opgeslagen!")

def view_history():
    st.header("Eerdere gesprekken")
    st.info("Hier komen je opgeslagen chats.")

def view_accountant():
    st.header("Deel met Accountant")
    if st.button("Genereer PDF Rapport"):
        st.success("Rapport verstuurd.")

# --- 8. HOOFDSTRUCTUUR ---

def main():
    st.sidebar.title("Finny 🤖")
    menu = st.sidebar.radio("Menu", ["Chat", "Kennismaking", "Eerdere gesprekken", "Deel met accountant"])
    
    if menu == "Chat":
        view_chat()
    elif menu == "Kennismaking":
        view_intro()
    elif menu == "Eerdere gesprekken":
        view_history()
    elif menu == "Deel met accountant":
        view_accountant()

if __name__ == "__main__":
    main()
