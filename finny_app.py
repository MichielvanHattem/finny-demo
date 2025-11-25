import streamlit as st
import pandas as pd
import os
# import fitz  # PyMuPDF - Uncomment als je PDF parsing lokaal hebt draaien
import re

# ==========================================
# 1. SETUP & CONFIGURATIE
# ==========================================
st.set_page_config(page_title="Finny Demo", page_icon="💰", layout="wide")

# Pad naar logo (pas aan als het pad anders is in jouw omgeving)
LOGO_PATH = "finny_logo.png"

# Initialiseer Session State variabelen als ze nog niet bestaan
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben Finny. Wat wil je weten over je cijfers?"}]
if "conversations" not in st.session_state:
    st.session_state.conversations = []  # Lijst van eerdere gesprekken
if "client_profile" not in st.session_state:
    st.session_state.client_profile = {}
if "current_view" not in st.session_state:
    st.session_state.current_view = "Chat"
if "fact_cache" not in st.session_state:
    st.session_state.fact_cache = {}  # Stap 3 voorbereiding: (metric, year) -> value

# ==========================================
# 2. DATA LOADING (ROBUUST GEMAAKT)
# ==========================================
@st.cache_data
def load_data():
    """
    Laadt de CSV-bestanden in DataFrames.
    Gebruikt error handling om crashes te voorkomen bij ontbrekende files.
    """
    data_files = {
        "trans": "Finny_Transactions.csv",
        "syn": "Finny_Synonyms.csv",
        "rgs": "Finny_RGS.csv"
    }
    
    dfs = {}
    
    for key, filename in data_files.items():
        try:
            if os.path.exists(filename):
                # engine='python' helpt vaak bij separator issues, sep=None laat pandas gokken (, of ;)
                dfs[key] = pd.read_csv(filename, sep=None, engine='python')
                
                # Normaliseer kolomnamen naar lowercase en strip spaties voor veiligheid
                dfs[key].columns = dfs[key].columns.str.strip().str.lower()
            else:
                st.warning(f"Bestand niet gevonden: {filename}. Functionaliteit kan beperkt zijn.")
                dfs[key] = pd.DataFrame()
        except Exception as e:
            st.error(f"Fout bij laden {filename}: {e}")
            dfs[key] = pd.DataFrame()
            
    return dfs["trans"], dfs["syn"], dfs["rgs"]

# Laad de data
df_trans, df_syn, df_rgs = load_data()

# ==========================================
# 3. HELPER FUNCTIES & LOGICA
# ==========================================

def get_column_flexible(df, candidates):
    """
    Zoekt in de dataframe kolommen naar een match uit de kandidatenlijst.
    Dit voorkomt KeyError als de kolom 'Keyword' ineens 'Synoniem' heet.
    """
    for col in df.columns:
        if col in candidates:
            return col
    return None

def determine_intent(user_input, df_syn):
    """
    Bepaalt of de vraag over PDF (jaarrekening) of CSV (transacties) gaat.
    Fix: Robuuste kolom-check om KeyError te voorkomen.
    """
    user_input_lower = user_input.lower()
    
    # Default intent
    intent = "PDF" 
    category_found = None
    
    if df_syn.empty:
        return intent, category_found

    # 1. Zoek de relevante kolomnamen (want die kunnen variëren in de CSV)
    # We zoeken naar kolommen als: keyword, synoniem, trefwoord, term
    keyword_col = get_column_flexible(df_syn, ['keyword', 'synoniem', 'synonym', 'trefwoord', 'term'])
    category_col = get_column_flexible(df_syn, ['category', 'categorie', 'rubriek', 'finny_category'])

    # Als we geen keyword kolom kunnen vinden, vallen we terug op PDF (veiligheid)
    if not keyword_col:
        # Optioneel: print warning in console, niet in UI om gebruiker niet te storen
        print("Warning: Geen keyword-kolom gevonden in synonymen file.")
        return "PDF", None

    # 2. Loop door de synoniemen
    for _, row in df_syn.iterrows():
        # Veilig converteren naar string en lowercase
        keyword = str(row[keyword_col]).lower().strip()
        
        if keyword and keyword in user_input_lower:
            intent = "CSV"
            if category_col:
                category_found = row[category_col]
            break
            
    return intent, category_found

def calculate_csv_answer(year, category, df_trans):
    """
    Berekent totalen uit CSV op basis van jaar en categorie.
    """
    if df_trans.empty:
        return "Ik heb geen transactiedata geladen."

    # Zorg dat datum kolom herkend wordt (zoek naar 'date', 'datum', of eerste kolom)
    date_col = get_column_flexible(df_trans, ['date', 'datum', 'transactiedatum'])
    amount_col = get_column_flexible(df_trans, ['amount', 'bedrag', 'waarde'])
    cat_col = get_column_flexible(df_trans, ['category', 'categorie', 'grootboek', 'rubriek'])

    if not (date_col and amount_col):
        return "Ik kan de datum of bedrag kolommen niet vinden in de transacties."

    # Filter op jaar
    # We nemen aan dat datum strings zijn of datetime objects.
    try:
        # Converteer voor zekerheid naar datetime
        df_trans['temp_date'] = pd.to_datetime(df_trans[date_col], errors='coerce')
        df_filtered = df_trans[df_trans['temp_date'].dt.year == int(year)]
    except:
        # Fallback als datums niet parsen: string match op jaar
        df_filtered = df_trans[df_trans[date_col].astype(str).str.contains(str(year), na=False)]

    if df_filtered.empty:
        return f"Ik heb geen transacties gevonden voor {year}."

    # Filter op categorie (indien bekend en kolom bestaat)
    if category and cat_col:
        # Simpele case-insensitive match
        df_filtered = df_filtered[df_filtered[cat_col].astype(str).str.lower() == str(category).lower()]
    
    # Bereken som
    total = df_filtered[amount_col].sum()
    return f"De totale {category if category else 'kosten'} in {year} bedragen € {total:,.2f}."

def get_pdf_answer_mock(user_input):
    """
    Simuleert de PDF antwoorden voor de stabiliteitstest.
    Vervang dit met je echte 'query_engine' of PDF-logica.
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
    # 1. Bepaal jaar (simpele extractie)
    year = "2024" # Default
    years = re.findall(r'202[2-4]', user_input)
    if years:
        year = years[0]

    # 2. Bepaal Intent
    intent, category = determine_intent(user_input, df_syn)
    
    # 3. Genereer antwoord
    response = ""
    
    if intent == "CSV":
        response = calculate_csv_answer(year, category, df_trans)
    else:
        # Hier roep je normaal je PDF/RAG functie aan
        response = get_pdf_answer_mock(user_input)

    return response

def save_current_conversation():
    """Slaat huidige berichten op in historie en wist chat."""
    if len(st.session_state.messages) > 1: # Alleen als er echt gepraat is
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
        st.title("Finny") # Fallback als plaatje mist
    
    st.markdown("---")
    
    # MENU
    menu_choice = st.radio("Menu", ["Chat", "Kennismaking", "Eerdere gesprekken", "Deel met accountant"])
    
    # UPDATE VIEW STATE
    if menu_choice != st.session_state.current_view:
        st.session_state.current_view = menu_choice
        st.rerun()
        
    st.markdown("---")
    st.info(f"Geheugen: [2022, 2023, 2024]\nData status: {'🟢' if not df_trans.empty else '🔴'}")

    # Nieuw gesprek knop (alleen in chat view relevant, maar mag altijd)
    if st.button("Nieuw gesprek"):
        save_current_conversation()
        st.rerun()

# ==========================================
# 5. MAIN VIEWS
# ==========================================

# --- VIEW: CHAT ---
if st.session_state.current_view == "Chat":
    st.title("Finny Demo - Chat")

    # Toon historie
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Stel je vraag over je cijfers..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant logic
        with st.spinner("Finny denkt na..."):
            answer = handle_user_input(prompt)
        
        # Assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# --- VIEW: KENNISMAKING ---
elif st.session_state.current_view == "Kennismaking":
    st.title("Kennismaking")
    st.write("Vertel ons iets meer over je bedrijf.")
    
    with st.form("profile_form"):
        comp_name = st.text_input("Bedrijfsnaam", value=st.session_state.client_profile.get("name", ""))
        risk_appetite = st.slider("Risicobereidheid", 0, 10, value=st.session_state.client_profile.get("risk", 5))
        sector = st.selectbox("Sector", ["IT", "Bouw", "Horeca", "Overig"], index=0)
        
        submitted = st.form_submit_button("Opslaan")
        if submitted:
            st.session_state.client_profile = {
                "name": comp_name,
                "risk": risk_appetite,
                "sector": sector
            }
            st.success("Profiel opgeslagen!")

# --- VIEW: EERDERE GESPREKKEN ---
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

# --- VIEW: DEEL MET ACCOUNTANT ---
elif st.session_state.current_view == "Deel met accountant":
    st.title("Deel met accountant")
    st.write("Selecteer gesprekken die je wilt delen.")
    
    if not st.session_state.conversations:
        st.write("Geen gesprekken om te delen.")
    else:
        for i, conv in enumerate(st.session_state.conversations):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                # Checkbox state koppelen aan interne data zou netter zijn, 
                # maar voor nu puur UI stabiliteit.
                checked = st.checkbox("", key=f"chk_{i}", value=conv["shared_with_accountant"])
                if checked != conv["shared_with_accountant"]:
                    conv["shared_with_accountant"] = checked
            with col2:
                st.write(f"Gesprek {conv['id']} - {conv['summary']}")
                if conv["shared_with_accountant"]:
                    st.caption("✅ Gemarkeerd om te delen")
