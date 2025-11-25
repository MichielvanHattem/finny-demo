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
    
    # FACT CACHE: De 'Single Source of Truth'
    # Slaat berekende antwoorden op zodat ze consistent blijven.
    # Structuur: {(metric, category, year): {'amount': 123.45, 'source': 'CSV', 'text': '...'}}
    if "fact_cache" not in st.session_state:
        st.session_state.fact_cache = {}
    
    # Context geheugen (onthoudt over welk jaar we praten)
    if "active_years" not in st.session_state:
        st.session_state.active_years = [datetime.now().year - 1] 
    
    # Client profiel (voor de kennismaking view)
    if "client_profile" not in st.session_state:
        st.session_state.client_profile = {}

init_state()

# --- 2. DATA LADEN (De Fix voor de Crash) ---

@st.cache_data
def load_data():
    """
    Laadt data direct uit CSV's met pandas.
    Dit voorkomt de 'ValueError: arrays must be same length'.
    """
    try:
        # Probeer echte bestanden te laden
        # Let op de dtypes (GLCode als string houden om voorloopnullen te bewaren indien nodig)
        df_trans = pd.read_csv(
            "Finny_Transactions.csv", 
            sep=";", 
            encoding="latin-1",
            dtype={'Finny_GLCode': str, 'Finny_Year': int}
        )
        
        # Bedragen converteren (komma naar punt indien nodig)
        if df_trans['AmountDC_num'].dtype == object:
            df_trans['AmountDC_num'] = df_trans['AmountDC_num'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

        df_syn = pd.read_csv("Finny_Synonyms.csv", sep=";", encoding="latin-1")
        df_rgs = pd.read_csv("Finny_RGS.csv", sep=";", encoding="latin-1", dtype={'Finny_GLCode': str})
        
        return df_trans, df_syn, df_rgs

    except FileNotFoundError:
        # FALLBACK: Mock data genereren als CSV's niet bestaan (zodat app niet crasht in demo)
        st.warning("⚠️ Geen CSV-bestanden gevonden. Finny draait in DEMO-modus met testdata.")
        
        # Veilige mock data die altijd werkt
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

# Data laden bij opstarten
df_trans, df_syn, df_rgs = load_data()

# --- 3. INTENT & ROUTER LOGICA ---

def extract_years(text):
    """Haalt jaartallen uit tekst en update sessie."""
    years = re.findall(r'\b(202[0-9])\b', text)
    if years:
        unique_years = sorted(list(set([int(y) for y in years])))
        st.session_state.active_years = unique_years
        return unique_years
    return st.session_state.active_years

def determine_intent(user_input):
    """
    De Verkeersregelaar:
    Bepaalt Bron (CSV of PDF) en Categorie op basis van keywords.
    """
    user_input_lower = user_input.lower()
    years = extract_years(user_input)
    
    # Keywords lijsten
    csv_keywords = ["telefoon", "gsm", "vodafone", "ziggo", "kpn", "transacties", "boekingen", "facturen", "kosten per", "leverancier", "top 10"]
    pdf_keywords = ["winst", "omzet", "resultaat", "totaal kosten", "jaarrekening", "balans", "passiva"]
    
    intent = {
        "source": "PDF", # Default fallback is PDF (veiliger voor algemene vragen)
        "metric": "general",
        "category": None,
        "years": years
    }
    
    # 1. Check CSV triggers (Heeft voorrang bij detailvragen)
    if any(kw in user_input_lower for kw in csv_keywords):
        intent["source"] = "CSV"
        intent["metric"] = "kosten"
        
        # Zoek categorie in synoniemen tabel
        found_cat = None
        for idx, row in df_syn.iterrows():
            if row['Keyword'] in user_input_lower:
                found_cat = row['Category']
                break
        
        if found_cat:
            intent["category"] = found_cat
        else:
            intent["category"] = "general_transactions"

    # 2. Check PDF triggers (Alleen als CSV geen duidelijke match heeft, of expliciet PDF termen)
    elif any(kw in user_input_lower for kw in pdf_keywords):
        intent["source"] = "PDF"
        if "winst" in user_input_lower: intent["metric"] = "winst"
        if "omzet" in user_input_lower: intent["metric"] = "omzet"
    
    return intent

# --- 4. CALCULATIE ENGINE (CSV) ---

def get_csv_metrics(intent):
    """
    Deterministische berekening op basis van CSV data.
    Gebruikt fact_cache om consistentie te garanderen.
    """
    category = intent["category"]
    year = intent["years"][0] # We pakken het eerste jaar voor de eenvoud
    metric = intent["metric"]
    
    # STAP 1: CHECK FACT CACHE
    cache_key = (metric, category, year)
    if cache_key in st.session_state.fact_cache:
        cached_data = st.session_state.fact_cache[cache_key]
        return f"FACT (uit geheugen): {cached_data['text']}"

    # STAP 2: BEREKENEN (Als niet in cache)
    if category == "general_transactions":
        return "CONTEXT: Ik zie dat je transacties wilt zien, maar ik kan de specifieke categorie niet bepalen. Vraag bijvoorbeeld specifiek naar 'telefoonkosten' of 'autokosten'."
    
    # 2a. Vind GL Codes via RGS tabel
    target_gl_codes = df_rgs[df_rgs['Category'] == category]['Finny_GLCode'].tolist()
    
    if not target_gl_codes:
        msg = f"In de administratie is geen specifieke grootboekrekening gekoppeld aan de categorie '{category}'. Ik kan daarom geen betrouwbaar totaalbedrag geven."
        st.session_state.fact_cache[cache_key] = {'amount': 0, 'source': 'CSV', 'text': msg}
        return f"FACT: {msg}"

    # 2b. Filter Transacties
    mask_year = df_trans['Finny_Year'] == year
    # Zorg voor string comparison voor GL codes
    mask_gl = df_trans['Finny_GLCode'].astype(str).isin([str(c) for c in target_gl_codes])
    
    filtered_df = df_trans[mask_year & mask_gl]
    
    total_amount = filtered_df['AmountDC_num'].sum()
    count = len(filtered_df)
    
    # 2c. Tekst bouwen
    if count > 0:
        fact_text = f"De totale {category} in {year} bedragen € {total_amount:,.2f}. Dit is de som van {count} boekingen op de grootboekrekeningen voor {category}."
    else:
        fact_text = f"Er zijn in {year} geen boekingen gevonden voor de categorie {category}."

    # STAP 3: OPSLAAN IN CACHE
    st.session_state.fact_cache[cache_key] = {
        'amount': total_amount,
        'source': 'CSV',
        'text': fact_text
    }
    
    return f"FACT (Nieuw berekend): {fact_text}"

# --- 5. PDF ENGINE (HIER JOUW OUDE CODE INVOEGEN) ---

def get_pdf_metrics(intent):
    """
    Haalt info uit de PDF/Vectorstore.
    """
    user_query = f"{intent['metric']} {intent['years'][0]}"
    
    # ------------------------------------------------------------------
    # [STRUCTUUR VOOR JOUW ORIGINELE CODE]
    # Plak hieronder jouw werkende LangChain/PDF code uit versie 1.0.
    # Als je die niet plakt, gebruikt hij de fallback hieronder.
    # ------------------------------------------------------------------
    
    try:
        # VOORBEELD (vervang dit door jouw echte aanroep):
        # response = your_qa_chain.run(user_query)
        # return f"CONTEXT UIT PDF: {response}"
        
        # Zolang je code er niet staat, doen we een mock voor de demo:
        year = intent["years"][0]
        if intent["metric"] == "winst":
            return f"FACT: In {year} was de winst na belastingen € 76.226 (volgens de jaarrekening PDF)."
        elif intent["metric"] == "omzet":
            return f"FACT: De omzet in {year} bedroeg € 120.000 (volgens de jaarrekening PDF)."
        else:
            return "CONTEXT: Ik heb gezocht in de PDF, maar vond geen exact antwoord op deze specifieke vraag."
            
    except Exception as e:
        return f"FOUT BIJ PDF ZOEKEN: {str(e)}"

# --- 6. LLM INTEGRATIE (ANTWOORD GENERATIE) ---

def generate_llm_response(intent, context_text):
    """
    Genereert het uiteindelijke antwoord.
    In een echte app roep je hier OpenAI aan.
    """
    
    # De System Prompt die hallucinaties voorkomt
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
    # We simuleren dat het LLM het feit netjes verwoordt.
    if "FACT" in context_text:
        # Haal de tekst na "FACT (...):" op
        clean_fact = context_text.split("):")[-1].strip()
        if intent["source"] == "CSV":
            return f"{clean_fact}" # De CSV functie geeft al een hele mooie zin terug
        else:
            return f"{clean_fact} (Gevonden in de jaarrekening)."
    else:
        return "Ik heb gezocht in de documenten, maar kon geen specifiek bedrag vinden voor je vraag."

# --- 7. UI VIEWS ---

def view_chat():
    st.header("Chat met Finny")
    
    # Toon historie
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if prompt := st.chat_input("Stel je vraag (bijv. 'Wat zijn mijn telefoonkosten 2024?')..."):
        # 1. Toon user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2. Bepaal Intent
        intent = determine_intent(prompt)
        
        # 3. Haal Context (CSV of PDF)
        if intent["source"] == "CSV":
            context = get_csv_metrics(intent)
        else:
            context = get_pdf_metrics(intent)
            
        # 4. Genereer Antwoord
        response = generate_llm_response(intent, context)

        # 5. Toon assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
            
        # Debug Expandertje (voor jou als developer)
        with st.expander("🕵️ Debug Info (Intent & Cache)"):
            st.write(f"**Intent:** {intent}")
            st.write(f"**Context:** {context}")
            st.write("**Fact Cache Inhoud:**", st.session_state.fact_cache)

def view_intro():
    st.header("Kennismaking met Finny")
    st.write("Vul je gegevens in zodat Finny je beter leert kennen.")
    
    with st.form("profile_form"):
        name = st.text_input("Je naam", value=st.session_state.client_profile.get("name", ""))
        role = st.selectbox("Je rol", ["DGA", "Boekhouder", "Investeerder"], index=0)
        focus = st.multiselect("Waar let je op?", ["Winst", "Kostenbeheersing", "Groei"], default=["Winst"])
        
        submitted = st.form_submit_button("Opslaan")
        if submitted:
            st.session_state.client_profile = {
                "name": name,
                "role": role,
                "focus": focus
            }
            st.success("Profiel opgeslagen! Je kunt nu naar de chat.")

    if st.session_state.client_profile:
        st.info(f"Huidig profiel: {st.session_state.client_profile}")

def view_history():
    st.header("Eerdere gesprekken")
    st.info("Hier komen je opgeslagen chats te staan (feature in ontwikkeling).")

def view_accountant():
    st.header("Deel met Accountant")
    st.write("Genereer hier een rapport voor je accountant.")
    if st.button("Genereer PDF Rapport"):
        st.success("Rapport verstuurd naar accountant@example.com (simulatie).")

# --- 8. HOOFDSTRUCTUUR (SIDEBAR) ---

def main():
    st.sidebar.title("Finny 🤖")
    
    # Sidebar menu
    menu = st.sidebar.radio(
        "Menu", 
        ["Chat", "Kennismaking", "Eerdere gesprekken", "Deel met accountant"]
    )
    
    # View router
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
