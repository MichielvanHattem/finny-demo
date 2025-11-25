import streamlit as st
import pandas as pd
import os
import re

# ==========================================
# 1. SETUP & CONFIGURATIE
# ==========================================
st.set_page_config(page_title="Finny Demo", page_icon="💰", layout="wide")

# Pad naar logo
LOGO_PATH = "finny_logo.png"

# Initialiseer Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben Finny. Wat wil je weten over je cijfers?"}]
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "client_profile" not in st.session_state:
    st.session_state.client_profile = {}
if "current_view" not in st.session_state:
    st.session_state.current_view = "Chat"
# Stap 4: Fact cache voor consistentie
if "fact_cache" not in st.session_state:
    st.session_state.fact_cache = {}

# ==========================================
# 2. DATA LOADING (STABIEL & SIMPEL)
# ==========================================
@st.cache_data
def load_data():
    """
    Laadt de CSV-bestanden.
    Forceert sep=";" en encoding="latin1" voor stabiliteit met NL Excel/CSV bestanden.
    """
    def read_csv_safe(path):
        if not os.path.exists(path):
            st.warning(f"Bestand niet gevonden: {path}")
            return pd.DataFrame()
        try:
            # Harde, veilige settings voor Nederlandse CSV's
            df = pd.read_csv(path, sep=";", encoding="latin1")
            # Normaliseer kolomnamen: strip spaties en maak lowercase
            df.columns = df.columns.str.strip().str.lower()
            return df
        except Exception as e:
            st.error(f"Fout bij laden {path}: {e}")
            return pd.DataFrame()

    df_trans = read_csv_safe("Finny_Transactions.csv")
    df_syn   = read_csv_safe("Finny_Synonyms.csv")
    df_rgs   = read_csv_safe("Finny_RGS.csv")
    
    return df_trans, df_syn, df_rgs

# Laad de data direct
df_trans, df_syn, df_rgs = load_data()

# ==========================================
# 3. HELPER FUNCTIES & LOGICA
# ==========================================

def get_column_flexible(df, candidates):
    """
    Zoekt in de dataframe kolommen naar een match uit de kandidatenlijst.
    """
    if df.empty:
        return None
    for col in df.columns:
        if col in candidates:
            return col
    return None

def determine_intent(user_input, df_syn):
    """
    Bepaalt of de vraag naar PDF (jaarrekening) of CSV (transacties) moet.
    Stap 2: Harde regels voor winst/omzet gaan VÓÓR synoniemen.
    """
    user_input_lower = user_input.lower()
    
    # 1. Harde regels voor Jaarrekening begrippen
    pdf_triggers = ["winst", "resultaat", "omzet", "bedrijfsopbrengst", "totale kosten"]
    csv_force_triggers = ["transacties", "boekingen", "grootboek", "telefoon", "communicatie", "vodafone", "ziggo", "auto", "brandstof"]
    
    # Als het een winst/omzet vraag is, EN er wordt niet expliciet om boekingen/details gevraagd:
    is_pdf_question = any(word in user_input_lower for word in pdf_triggers)
    is_csv_force = any(word in user_input_lower for word in csv_force_triggers)
    
    if is_pdf_question and not is_csv_force:
        return "PDF", None

    # 2. Synoniemen check voor CSV
    intent = "PDF" # Default fallback
    category_found = None
    
    if df_syn.empty:
        return intent, category_found

    keyword_col = get_column_flexible(df_syn, ['keyword', 'synoniem', 'synonym', 'trefwoord', 'term'])
    category_col = get_column_flexible(df_syn, ['category', 'categorie', 'rubriek', 'finny_category'])

    if not keyword_col:
        # Geen keyword kolom gevonden, veilig terugvallen op PDF
        return "PDF", None

    for _, row in df_syn.iterrows():
        val = row[keyword_col]
        if pd.isna(val):
            continue
            
        keyword = str(val).lower().strip()
        # Check op match
        if keyword and keyword in user_input_lower:
            intent = "CSV"
            if category_col and not pd.isna(row[category_col]):
                category_found = row[category_col]
            break
            
    return intent, category_found

def calculate_csv_answer(year, category, df_trans):
    """
    Stap 3: CSV berekening met specifieke Finny-kolommen.
    """
    if df_trans.empty:
        return "Ik heb geen transactiedata geladen."

    # 3.1 Uitgebreide kolomdetectie
    # Jaar: zoek ook naar 'finny_year' of 'jaar'
    year_col = get_column_flexible(df_trans, ['finny_year', 'jaar', 'date', 'datum', 'transactiedatum'])
    # Bedrag: zoek ook naar 'amountdc_num' (specifiek Finny)
    amount_col = get_column_flexible(df_trans, ['amountdc_num', 'amount', 'bedrag', 'waarde', 'amountdc'])
    # Categorie
    cat_col = get_column_flexible(df_trans, ['finny_category', 'finny_gldescription', 'category', 'categorie', 'rubriek'])

    if not (year_col and amount_col):
        # Wees specifiek in de foutmelding, maar crash niet.
        return "Ik kan geen jaar- of bedragkolom vinden in de transacties (check Finny_Transactions.csv)."

    # 3.2 Filteren
    try:
        df_filtered = df_trans.copy()
        
        # Jaar filteren: robuust voor int vs string
        df_filtered['str_year'] = df_filtered[year_col].astype(str)
        df_filtered = df_filtered[df_filtered['str_year'].str.contains(str(year), na=False)]
        
        if df_filtered.empty:
            return f"Ik heb geen transacties gevonden voor het jaar {year}."

        # Categorie filteren (indien bekend)
        cat_name_display = category if category else "kosten"
        if category and cat_col:
            df_filtered = df_filtered[df_filtered[cat_col].astype(str).str.lower() == str(category).lower()]
            if df_filtered.empty:
                return f"Ik zie geen transacties in {year} voor de categorie '{category}'."
        
        # 3.3 Som berekenen
        # Zorg dat bedrag numeriek is (komma's naar punten indien string)
        if df_filtered[amount_col].dtype == object:
            df_filtered[amount_col] = df_filtered[amount_col].str.replace(',', '.', regex=False)
        
        total = pd.to_numeric(df_filtered[amount_col], errors='coerce').sum()
        
        return f"De totale {cat_name_display} in {year} bedragen € {total:,.2f}."
        
    except Exception as e:
        return f"Er ging iets mis bij het berekenen: {e}"

def get_pdf_answer_mock(user_input):
    """
    Placeholder voor PDF logica (Simuleert Jaarrekening antwoorden).
    """
    user_input_lower = user_input.lower()
    
    # Winst
    if "winst" in user_input_lower or "resultaat" in user_input_lower:
        if "2024" in user_input_lower: return "Volgens de concept-jaarrekening is de winst in 2024: € 120.000."
        if "2023" in user_input_lower: return "Volgens de jaarrekening is de winst in 2023: € 100.000."
        if "2022" in user_input_lower: return "Volgens de jaarrekening is de winst in 2022: € 85.000."
        return "Over welk jaar wil je de winst weten?"
    
    # Omzet
    if "omzet" in user_input_lower:
        if "2024" in user_input_lower: return "De omzet in 2024 bedraagt: € 450.000."
        if "2023" in user_input_lower: return "De omzet in 2023 bedroeg: € 400.000."
        return "De omzet staat in de jaarrekening."
        
    # Totale kosten
    if "kosten" in user_input_lower:
         if "2024" in user_input_lower: return "De totale bedrijfskosten in 2024 waren € 330.000."

    return "Dat kan ik niet direct in de PDF vinden (Demo modus)."

def handle_user_input(user_input):
    # 1. Bepaal jaar (simpele extractie)
    year = "2024" # Default
    years = re.findall(r'202[2-4]', user_input)
    if years:
        year = years[0]

    # 2. Bepaal Intent en Categorie
    intent, category = determine_intent(user_input, df_syn)
    
    # Stap 4: Check Fact Cache (simpel)
    # Sleutel: (intent, category_or_keyword, year)
    # We gebruiken een vereenvoudigde sleutel voor deze demo
    cache_key = f"{intent}_{category}_{year}_{'winst' if 'winst' in user_input.lower() else 'overig'}"
    
    if cache_key in st.session_state.fact_cache:
        # Return gecached antwoord (kleine optimalisatie)
        return st.session_state.fact_cache[cache_key]

    # 3. Genereer antwoord
    response = ""
    if intent == "CSV":
        response = calculate_csv_answer(year, category, df_trans)
    else:
        response = get_pdf_answer_mock(user_input)
    
    # Sla op in cache
    st.session_state.fact_cache[cache_key] = response
    
    return response

def save_current_conversation():
    if len(st.session_state.messages) > 1:
        summary = st.session_state.messages[-1]['content'][:50] + "..."
        st.session_state.conversations.append({
            "id": len(st.session_state.conversations) + 1,
            "summary": summary,
            "messages": st.session_state.messages,
            "shared_with_accountant": False
        })
    st.session_state.messages = [{"role": "assistant", "content": "Nieuw gesprek gestart. Zeg het maar!"}]

# ==========================================
# 4. SIDEBAR & NAVIGATIE
# ==========================================
with st.sidebar:
    # LOGO
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=150)
    else:
        st.title("Finny Demo")
    
    st.markdown("---")
    
    # MENU
    menu_choice = st.radio("Menu", ["Chat", "Kennismaking", "Eerdere gesprekken", "Deel met accountant"])
    
    # Update view state
    if menu_choice != st.session_state.current_view:
        st.session_state.current_view = menu_choice
        st.rerun()
        
    st.markdown("---")
    
    # Status indicator
    data_loaded = not df_trans.empty
    st.info(f"Geheugen: [2022, 2023, 2024]\nData status: {'🟢' if data_loaded else '🔴'}")

    if st.button("Nieuw gesprek"):
        save_current_conversation()
        st.rerun()

# ==========================================
# 5. MAIN VIEWS
# ==========================================

# --- CHAT ---
if st.session_state.current_view == "Chat":
    st.title("Finny Demo - Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Stel je vraag over je cijfers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Finny denkt na..."):
            answer = handle_user_input(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# --- KENNISMAKING ---
elif st.session_state.current_view == "Kennismaking":
    st.title("Kennismaking")
    st.write("Vertel ons iets meer over je bedrijf.")
    
    with st.form("profile_form"):
        comp_name = st.text_input("Bedrijfsnaam", value=st.session_state.client_profile.get("name", ""))
        risk_appetite = st.slider("Risicobereidheid", 0, 10, value=st.session_state.client_profile.get("risk", 5))
        sector = st.selectbox("Sector", ["IT", "Bouw", "Horeca", "Overig"], index=0)
        
        if st.form_submit_button("Opslaan"):
            st.session_state.client_profile = {
                "name": comp_name,
                "risk": risk_appetite,
                "sector": sector
            }
            st.success("Profiel opgeslagen!")

# --- EERDERE GESPREKKEN ---
elif st.session_state.current_view == "Eerdere gesprekken":
    st.title("Eerdere gesprekken")
    if not st.session_state.conversations:
        st.write("Nog geen gesprekken opgeslagen.")
    else:
        for conv in st.session_state.conversations:
            st.markdown(f"**Gesprek {conv['id']}**: {conv['summary']}")
            with st.expander("Bekijk details"):
                for m in conv['messages']:
                    st.write(f"**{m['role']}**: {m['content']}")

# --- DEEL MET ACCOUNTANT ---
elif st.session_state.current_view == "Deel met accountant":
    st.title("Deel met accountant")
    if not st.session_state.conversations:
        st.write("Geen gesprekken om te delen.")
    else:
        for i, conv in enumerate(st.session_state.conversations):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                checked = st.checkbox("", key=f"chk_{i}", value=conv["shared_with_accountant"])
                if checked != conv["shared_with_accountant"]:
                    conv["shared_with_accountant"] = checked
            with col2:
                st.write(f"Gesprek {conv['id']} - {conv['summary']}")
                if conv["shared_with_accountant"]:
                    st.caption("✅ Gemarkeerd om te delen")
