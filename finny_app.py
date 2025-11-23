import streamlit as st
import pandas as pd
import re

# ------------------------------------------------------------------
# 1. CONFIGURATIE & SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Finny AI", page_icon="🦁", layout="wide")

# Forceer dat kolommen die codes bevatten als tekst worden gelezen
DTYPE_SETTINGS = {
    'Finny_GLCode': str,
    'RGS_Referentiecode': str,
    'AccountCode_norm': str
}

# ------------------------------------------------------------------
# 2. DATA LADEN (De 3 nieuwe bronnen)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Laad Transacties
        df_trans = pd.read_csv('Finny_Transactions.csv', sep=';', dtype=DTYPE_SETTINGS)
        # Zorg dat getallen floats zijn (punt als decimaal)
        if df_trans['AmountDC_num'].dtype == object:
             df_trans['AmountDC_num'] = pd.to_numeric(df_trans['AmountDC_num'].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Laad Synoniemen
        df_syn = pd.read_csv('Finny_Synonyms.csv', sep=';', dtype=DTYPE_SETTINGS)
        # Zet alles naar lowercase voor slim zoeken
        df_syn['Synoniem_lower'] = df_syn['Synoniem'].astype(str).str.lower()

        # Laad RGS (voor context/uitleg)
        df_rgs = pd.read_csv('Finny_RGS.csv', sep=';', dtype=DTYPE_SETTINGS)

        return df_trans, df_syn, df_rgs
    except Exception as e:
        st.error(f"Fout bij laden data: {e}")
        return None, None, None

df_trans, df_syn, df_rgs = load_data()

# ------------------------------------------------------------------
# 3. LOGICA: HET 3-STAPPEN PLAN
# ------------------------------------------------------------------

def extract_years(question):
    """Haalt jaartallen uit de vraag (bv. 2022, 2023)."""
    # Zoek naar reeksen van 4 cijfers die beginnen met 20
    years = re.findall(r'\b(20\d{2})\b', question)
    return [int(y) for y in years]

def find_gl_codes(question, df_syn, df_rgs):
    """Stap 1: Vertaal gebruikersvraag naar GL Codes."""
    question_lower = question.lower()
    found_codes = set()
    debug_matches = []

    # 1A. Check Synoniemenlijst
    for index, row in df_syn.iterrows():
        synoniem = row['Synoniem_lower']
        if synoniem in question_lower:
            found_codes.add(row['Finny_GLCode'])
            debug_matches.append(f"Synoniem '{row['Synoniem']}' -> GL {row['Finny_GLCode']}")
    
    # 1B. Als back-up: zoek in RGS omschrijvingen
    if not found_codes:
        for index, row in df_rgs.iterrows():
            # Check OmschrijvingKort en Omschrijving
            omschrijving = str(row.get('RGS_Omschrijving', '')).lower()
            if omschrijving and omschrijving in question_lower:
                found_codes.add(row['Finny_GLCode'])
                debug_matches.append(f"RGS tekst match '{row['RGS_Omschrijving']}' -> GL {row['Finny_GLCode']}")

    return list(found_codes), debug_matches

def get_financial_answer(question, df_trans, gl_codes):
    """Stap 2 & 3: Filter transacties en bereken totaal."""
    
    # Stap 2: Filter op GL Code
    filtered_df = df_trans[df_trans['Finny_GLCode'].isin(gl_codes)].copy()
    
    # Filter op Jaren (indien genoemd in vraag)
    years = extract_years(question)
    if years:
        # Als er jaartallen zijn, filteren we daarop
        filtered_df = filtered_df[filtered_df['Finny_Year'].isin(years)]
        year_str = ", ".join(map(str, years))
    else:
        # Geen jaar? Pak standaard het laatste jaar of alles (hier kiezen we alles voor totaalbeeld)
        year_str = "alle beschikbare jaren"

    # Stap 3: Aggregatie
    total_amount = filtered_df['AmountDC_num'].sum()
    
    # Groeperen per jaar voor detail
    if not filtered_df.empty:
        per_year = filtered_df.groupby('Finny_Year')['AmountDC_num'].sum().to_dict()
    else:
        per_year = {}

    return total_amount, per_year, year_str, filtered_df

# ------------------------------------------------------------------
# 4. DE INTERFACE
# ------------------------------------------------------------------

st.title("🦁 Finny 2.0")
st.markdown("Ik ben direct gekoppeld aan je RGS en transacties. Vraag maar raak.")

# Initialiseer chat geschiedenis
if "messages" not in st.session_state:
    st.session_state.messages = []

# Toon eerdere berichten
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Bijv: Wat waren de telefoonkosten in 2023?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # --- HET DENKPROCES ---
    if df_trans is not None:
        # Stap 1: Zoek GL codes
        gl_codes, debug_info = find_gl_codes(prompt, df_syn, df_rgs)
        
        if not gl_codes:
            response = "😕 Ik kon geen categorie vinden in je vraag. Probeer termen te gebruiken die in je synoniemenlijst staan (zoals 'autokosten', 'telefoon', etc)."
        else:
            # Stap 2 & 3: Rekenwerk
            total, per_year, period_tekst, detail_df = get_financial_answer(prompt, df_trans, gl_codes)
            
            # Formatteer het antwoord
            # Haal de naam van de rekening op (pak de eerste match uit RGS of Synoniemen)
            rekening_naam = "Geselecteerde kosten"
            if not detail_df.empty:
                rekening_naam = detail_df.iloc[0]['Finny_GLDescription']
            
            match_info = f"(Gevonden via codes: {', '.join(gl_codes)})"
            
            # Bouw de tekst op
            response = f"**{rekening_naam}** in {period_tekst}:\n\n"
            response += f"# € {total:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + "\n\n"
            
            if per_year:
                response += "**Verdeling per jaar:**\n"
                for year, amount in per_year.items():
                    formatted = f"€ {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    response += f"* {year}: {formatted}\n"

            # Voeg debug info toe in een 'expander' (inklapbaar)
            with st.expander("🔍 Hoe ik dit berekend heb"):
                st.write("Gevonden matches:", debug_info)
                st.write(f"Aantal transacties gevonden: {len(detail_df)}")
                if len(detail_df) > 0:
                    st.dataframe(detail_df[['EntryDate', 'Finny_GLDescription', 'AmountDC_num', 'Description']].head(5))

    else:
        response = "⚠️ De data kon niet geladen worden. Check de CSV bestanden."

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
