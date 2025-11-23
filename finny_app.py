import streamlit as st
import pandas as pd
import re
import os
import glob
import pdfplumber

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
# 2. DATA LADEN (CSV's)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Laad Transacties
        df_trans = pd.read_csv('Finny_Transactions.csv', sep=';', dtype=DTYPE_SETTINGS)
        # Fix: zet komma-getallen om naar punt-getallen (floats)
        if df_trans['AmountDC_num'].dtype == object:
             df_trans['AmountDC_num'] = pd.to_numeric(df_trans['AmountDC_num'].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Laad Synoniemen & RGS
        df_syn = pd.read_csv('Finny_Synonyms.csv', sep=';', dtype=DTYPE_SETTINGS)
        df_syn['Synoniem_lower'] = df_syn['Synoniem'].astype(str).str.lower()
        
        df_rgs = pd.read_csv('Finny_RGS.csv', sep=';', dtype=DTYPE_SETTINGS)

        return df_trans, df_syn, df_rgs
    except Exception as e:
        st.error(f"Fout bij laden CSV data: {e}")
        return None, None, None

df_trans, df_syn, df_rgs = load_data()

# ------------------------------------------------------------------
# 3. PDF LOGICA (JAARREKENING CHECK)
# ------------------------------------------------------------------
def search_pdfs(query):
    """Zoekt simpel naar trefwoorden in PDF bestanden (jaarrekeningen)."""
    results = []
    # Zoek alle PDF's in de map
    pdf_files = glob.glob("*.pdf") 
    
    if not pdf_files:
        return None

    query_words = query.lower().split()
    stop_words = ['de', 'het', 'een', 'in', 'van', 'wat', 'zijn', 'kosten', 'hoeveel', 'finny']
    search_terms = [w for w in query_words if w not in stop_words and len(w) > 3]

    if not search_terms:
        return None

    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        text_lower = text.lower()
                        score = sum(1 for term in search_terms if term in text_lower)
                        if score > 0:
                            # Maak een leesbare snippet
                            snippet = "..."
                            for term in search_terms:
                                idx = text_lower.find(term)
                                if idx != -1:
                                    start = max(0, idx - 50)
                                    end = min(len(text), idx + 150)
                                    snippet = text[start:end].replace('\n', ' ')
                                    break 
                            
                            results.append({
                                'file': os.path.basename(pdf_file),
                                'page': page_num + 1,
                                'snippet': snippet,
                                'score': score
                            })
        except Exception:
            continue # Sla bestand over als het niet leesbaar is

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:3]

# ------------------------------------------------------------------
# 4. CSV LOGICA (HET 3-STAPPEN PLAN)
# ------------------------------------------------------------------
def extract_years(question):
    """Haalt jaartallen uit de vraag."""
    years = re.findall(r'\b(20\d{2})\b', question)
    return [int(y) for y in years]

def find_gl_codes(question, df_syn, df_rgs):
    """Vertaalt vraag naar GL codes via Synoniemen of RGS."""
    question_lower = question.lower()
    found_codes = set()
    debug_matches = []

    # 1A. Check Synoniemenlijst (Prioriteit)
    for index, row in df_syn.iterrows():
        if row['Synoniem_lower'] in question_lower:
            found_codes.add(row['Finny_GLCode'])
            debug_matches.append(f"Synoniem '{row['Synoniem']}' -> GL {row['Finny_GLCode']}")
    
    # 1B. Back-up: zoek in RGS omschrijvingen
    if not found_codes:
        for index, row in df_rgs.iterrows():
            omschrijving = str(row.get('RGS_Omschrijving', '')).lower()
            if omschrijving and omschrijving in question_lower:
                found_codes.add(row['Finny_GLCode'])
                debug_matches.append(f"RGS match '{row['RGS_Omschrijving']}' -> GL {row['Finny_GLCode']}")

    return list(found_codes), debug_matches

def get_financial_answer(question, df_trans, gl_codes):
    """Filtert transacties en berekent totalen."""
    # Stap 2: Filter op Code
    filtered_df = df_trans[df_trans['Finny_GLCode'].isin(gl_codes)].copy()
    
    # Stap 2B: Filter op Jaar (indien in vraag)
    years = extract_years(question)
    if years:
        filtered_df = filtered_df[filtered_df['Finny_Year'].isin(years)]
        year_str = ", ".join(map(str, years))
    else:
        year_str = "totaal (alle jaren)"

    # Stap 3: Sommeer
    total_amount = filtered_df['AmountDC_num'].sum()
    
    # Groepeer per jaar voor detail
    if not filtered_df.empty:
        per_year = filtered_df.groupby('Finny_Year')['AmountDC_num'].sum().to_dict()
    else:
        per_year = {}

    return total_amount, per_year, year_str, filtered_df

# ------------------------------------------------------------------
# 5. DE INTERFACE
# ------------------------------------------------------------------

# LOGO: Zoek dynamisch naar de eerste jpg of png in de map
col1, col2 = st.columns([1, 4])
with col1:
    # Dit pakt elk plaatje dat in je map staat, ongeacht de naam
    logo_files = glob.glob("*.jpg") + glob.glob("*.png")
    if logo_files:
        st.image(logo_files[0], width=120)
    else:
        st.write("🦁") 
        
with col2:
    st.title("Finny")
    st.markdown("**Financial Assistant v11** - *Powered by PDF & CSV*")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Toon geschiedenis
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Vraag Finny (bijv. 'Wat zijn de telefoonkosten?' of 'Wat zegt het jaarverslag?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # --- ANTWOORD GENEREREN ---
    full_response = ""
    
    # 1. SCAN PDF (Documenten)
    pdf_results = search_pdfs(prompt)
    if pdf_results:
        pdf_text = "**📄 Gevonden in documenten:**\n"
        for res in pdf_results:
            pdf_text += f"- *{res['file']} (p.{res['page']})*: \"...{res['snippet']}...\"\n"
        full_response += pdf_text + "\n---\n"
    
    # 2. SCAN CSV (Harde Cijfers)
    if df_trans is not None:
        gl_codes, debug_info = find_gl_codes(prompt, df_syn, df_rgs)
        
        if gl_codes:
            # We hebben een match in de boekhouding
            total, per_year, period_tekst, detail_df = get_financial_answer(prompt, df_trans, gl_codes)
            
            # Pak de naam van de rekening voor de display
            rekening_naam = detail_df.iloc[0]['Finny_GLDescription'] if not detail_df.empty else "Geselecteerde post"
            
            csv_text = f"**📊 Boekhouding (Transacties):**\n"
            # Format getal (NL notatie)
            total_fmt = f"€ {total:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            
            csv_text += f"Op **{rekening_naam}** zie ik in {period_tekst} een totaal van **{total_fmt}**.\n"
            
            if per_year and (len(per_year) > 1 or "per jaar" in prompt.lower()):
                csv_text += "\n**Verdeling per jaar:**\n"
                for year, amount in per_year.items():
                    fmt = f"€ {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    csv_text += f"- {year}: {fmt}\n"
            
            full_response += csv_text
        
        elif not pdf_results:
            # Geen PDF en geen CSV match
            full_response = "😕 Ik kan het antwoord niet vinden in de PDF's en herken ook geen categorie voor de boekhouding. Probeer een andere term."
        
        if not gl_codes and pdf_results:
            full_response += "\n*(Ik heb geen specifieke boekhoudtransacties gevonden voor deze vraag, dus ik baseer me op de tekst hierboven)*"

    else:
        full_response = "⚠️ CSV data kon niet geladen worden."

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.chat_message("assistant").write(full_response)
    
    # Debug info
    with st.expander("🔍 Finny's Brein"):
        if pdf_results:
            st.write("Gevonden in PDF:", pdf_results)
        if df_trans is not None and 'gl_codes' in locals() and gl_codes:
            st.write(f"Gekoppelde GL codes: {gl_codes}")
            st.write("Logica:", debug_info)
