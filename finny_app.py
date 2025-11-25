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

# Initialiseer Session State variabelen
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben Finny. Wat wil je weten over je cijfers?"}]
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "client_profile" not in st.session_state:
    st.session_state.client_profile = {}
if "current_view" not in st.session_state:
    st.session_state.current_view = "Chat"
if "fact_cache" not in st.session_state:
    st.session_state.fact_cache = {}

# ==========================================
# 2. DATA LOADING (MET ENCODING FIX)
# ==========================================
@st.cache_data
def load_data():
    """
    Laadt de CSV-bestanden in DataFrames.
    Bevat een fix voor UnicodeDecodeError door te proberen met 'latin1'
    als standaard 'utf-8' faalt.
    """
    data_files = {
        "trans": "Finny_Transactions.csv",
        "syn": "Finny_Synonyms.csv",
        "rgs": "Finny_RGS.csv"
    }
    
    dfs = {}
    
    for key, filename in data_files.items():
        # Begin met een lege DataFrame voor veiligheid
        dfs[key] = pd.DataFrame()
        
        if os.path.exists(filename):
            try:
                # POGING 1: Probeer standaard UTF-8
                # sep=None en engine='python' zorgen dat hij zelf zoekt naar ; of ,
                dfs[key] = pd.read_csv(filename, sep=None, engine='python')
            except UnicodeDecodeError:
                try:
                    # POGING 2: Probeer Latin-1 (voor Excel files met ë, é, €)
                    dfs[key] = pd.read_csv(filename, sep=None, engine='python', encoding='latin1')
                except Exception as e:
                    st.error(f"Kon {filename} niet lezen met Latin-1: {e}")
            except Exception as e:
                st.error(f"Fout bij laden {filename}: {e}")

            # Als laden gelukt is, normaliseer kolomnamen
            if not dfs[key].empty:
                dfs[key].columns = dfs[key].columns.str.strip().str.lower()
        else:
            # Bestand bestaat niet, geef waarschuwing maar crash niet
            st.warning(f"Bestand niet gevonden: {filename}. Functionaliteit beperkt.")
            
    return dfs["trans"], dfs["syn"], dfs["rgs"]

# Laad de data direct bij start
df_trans, df_syn, df_rgs = load_data()

# ==========================================
# 3. HELPER FUNCTIES & LOGICA
# ==========================================

def get_column_flexible(df, candidates):
    """
    Zoekt in de dataframe kolommen naar een match uit de kandidatenlijst.
    Voorkomt KeyError als de kolomnaam net anders is.
    """
    if df.empty:
        return None
    for col in df.columns:
        if col in candidates:
            return col
    return None

def determine_intent(user_input, df_syn):
    """
    Bepaalt of de vraag over PDF (jaarrekening) of CSV (transacties) gaat.
    Robuust gemaakt tegen ontbrekende kolommen.
    """
    user_input_lower = user_input.lower()
    intent = "PDF" # Default
    category_found = None
    
    if df_syn.empty:
        return intent, category_found

    # Zoek flexibel naar de kolomnamen
    keyword_col = get_column_flexible(df_syn, ['keyword', 'synoniem', 'synonym', 'trefwoord', 'term'])
    category_col = get_column_flexible(df_syn, ['category', 'categorie', 'rubriek', 'finny_category'])

    # Als we geen kolom hebben om in te zoeken, stop dan veilig
    if not keyword_col:
        return "PDF", None

    # Loop door synoniemen
    for _, row in df_syn.iterrows():
        val = row[keyword_col]
        # Check of waarde niet leeg/NaN is
        if pd.isna(val):
            continue
            
        keyword = str(val).lower().strip()
        if keyword and keyword in user_input_lower:
            intent = "CSV"
            if category_col and not pd.isna(row[category_col]):
                category_found = row[category_col]
            break
            
    return intent, category_found

def calculate_csv_answer(year, category, df_trans):
    """
    Berekent totalen uit CSV op basis van jaar en categorie.
    """
    if df_trans.empty:
        return "Ik heb geen transactiedata geladen."

    # Flexibele kolomnamen
    date_col = get_column_flexible(df_trans, ['date', 'datum', 'transactiedatum'])
    amount_col = get_column_flexible(df_trans, ['amount', 'bedrag', 'waarde'])
    cat_col = get_column_flexible(df_trans, ['category', 'categorie', 'grootboek', 'rubriek'])

    if not (date_col and amount_col):
        return "Ik mis essentiële kolommen (datum/bedrag) in de transacties."

    # Filter op jaar
    try:
        # Probeer datum te parsen
        temp_dates = pd.to_datetime(df_trans[date_col], errors='coerce')
        # Filter rijen waar jaar klopt
        df_filtered = df_trans[temp_dates.dt.year == int(year)]
    except:
        # Fallback: string match
        df_filtered = df_trans[df_trans[date_col].astype(str).str.contains(str(year), na=False)]

    if df_filtered.empty:
        return f"Ik heb geen transacties gevonden voor het jaar {year}."

    # Filter op categorie
    if category and cat_col:
        df_filtered = df_filtered[df_filtered[cat_col].astype(str).str.lower() == str(category).lower()]
    
    # Bereken som (zorg dat bedrag numeriek is)
    try:
        # Vervang eventuele komma's door punten als het strings zijn
        if df_filtered[amount_col].dtype == object:
            df_filtered[amount_col] = df_filtered[amount_col].str.replace(',', '.', regex=False)
        
        total = pd.to_numeric(df_filtered[amount_col], errors='coerce').sum()
    except Exception:
        total = 0

    return f"De totale {category if category else 'kosten'} in {year} bedragen € {total:,.2f}."

def get_pdf_answer_mock(user_input):
    """
    Placeholder voor PDF logica om stabiliteit te testen.
    """
    user_input_lower = user_input.lower()
    if "winst" in user_input_lower:
        if "2024" in user_input_lower: return "Volgens de PDF is de winst in 2024: € 120.000."
        if "2023" in user_input_lower: return "Volgens de PDF is de winst in 2023: € 100.000."
        return "Over welk jaar wil je de winst weten?"
    if "omzet" in user_input_lower:
        if "2024" in user_input_lower: return "Volgens de PDF is de omzet in 2024: € 450.000."
        return "De omzet staat in de jaarrekening."
    
    return "Dat kan ik niet direct in de PDF vinden (Demo modus)."

def handle_user_input(user_input):
    # 1. Bepaal jaar
    year = "2024" 
    years = re.findall(r'202[2-4]', user_input)
    if years:
        year = years[0]

    # 2. Bepaal Intent
    intent, category = determine_intent(user_input, df_syn)
    
    # 3. Genereer antwoord
    if intent == "CSV":
        response = calculate_csv_answer(year, category, df_trans)
    else:
        response = get_pdf_answer_mock(user_input)

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
        st.title("Finny")
    
    st.markdown("---")
    
    # MENU
    menu_choice = st.radio("Menu", ["Chat", "Kennismaking", "Eerdere gesprekken", "Deel met accountant"])
    
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
