import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import json
from datetime import datetime

# ==========================================
# 1. CONFIGURATIE & AUTHENTICATIE
# ==========================================
st.set_page_config(page_title="Finny 5.4 | Robust & Debug", page_icon="🦁", layout="wide")

def check_password():
    """Eenvoudige wachtwoordbeveiliging."""
    def password_entered():
        if st.session_state["password"] == "demo2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        if os.path.exists("finny_logo.jpg"):
            st.image("finny_logo.jpg", width=150)
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Finny Portal</h1>", unsafe_allow_html=True)
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Wachtwoord", type="password", on_change=password_entered, key="password")
        st.error("Onjuist wachtwoord")
        return False
    else:
        return True

# ==========================================
# 2. DATA LADEN (ROBUUST)
# ==========================================
@st.cache_data
def load_data():
    data = {"trans": None, "ledger": None, "syllabus": "", "pdf_text": "", "years_available": []}
    
    # 1. Transacties
    if os.path.exists("Finny_Transactions_Lite.csv"):
        try:
            # Lees CSV met punt-komma, forceer strings niet meteen, laat pandas raden
            df = pd.read_csv("Finny_Transactions_Lite.csv", sep=";", on_bad_lines='skip', low_memory=False)
            
            # 1. Opschonen kolomnamen (verwijder spaties voor en achter)
            df.columns = df.columns.str.strip()
            
            # 2. Datum fixen
            if 'EntryDate' in df.columns:
                df['EntryDate'] = pd.to_datetime(df['EntryDate'], errors='coerce')
            
            # 3. Beschikbare jaren opslaan (voor de prompt straks)
            if 'Finny_Year' in df.columns:
                # Zorg dat jaren integers zijn
                df['Finny_Year'] = pd.to_numeric(df['Finny_Year'], errors='coerce').fillna(0).astype(int)
                unique_years = sorted(df[df['Finny_Year'] > 0]['Finny_Year'].unique().tolist())
                data["years_available"] = unique_years
            
            data["trans"] = df
        except Exception as e:
            st.error(f"Fout bij laden transacties: {e}")

    # 2. Grootboekschema
    if os.path.exists("Finny_GL_Lite.csv"):
        try:
            df_l = pd.read_csv("Finny_GL_Lite.csv", sep=";", on_bad_lines='skip')
            df_l.columns = df_l.columns.str.strip()
            data["ledger"] = df_l
        except Exception as e:
            st.error(f"Fout bij laden grootboek: {e}")

    # 3. Syllabus
    if os.path.exists("Finny_syllabus_v9_4_with_B3_B4_B5.txt"):
        try:
            with open("Finny_syllabus_v9_4_with_B3_B4_B5.txt", "r", encoding="utf-8") as f:
                data["syllabus"] = f.read()
        except:
            pass
            
    # 4. Jaarrekeningen PDF
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for i, page in enumerate(reader.pages):
                if i < 15: text += page.extract_text()
            data["pdf_text"] += f"--- BRON: {pdf} ---\n{text[:8000]}\n\n"
        except:
            pass
            
    return data

# ==========================================
# 3. DE LOGICA (ROUTER & QUERY)
# ==========================================

def get_finny_plan(client, question, syllabus, available_years):
    """
    Stap 1: Vraag -> Plan.
    Nu geven we de BESCHIKBARE JAREN mee, zodat hij niet naar 2025 zoekt als die niet bestaat.
    """
    years_str = ", ".join(map(str, available_years)) if available_years else "Onbekend"
    
    system_prompt = f"""
    Je bent de controller Finny.
    
    DATA CONTEXT:
    - Beschikbare boekjaren in CSV: {years_str}. (Als de gebruiker geen jaar noemt, kies dan het meest recente beschikbare jaar uit deze lijst!).
    - Huidige datum: {datetime.now().strftime('%d-%m-%Y')}.
    
    SYLLABUS (Synoniemen):
    {syllabus[:4000]} 
    
    TAAK:
    Bepaal zoekstrategie.
    
    OUTPUT (JSON):
    {{
        "source": "PDF" (voor algemene jaarcijfers/tekst), "CSV" (voor details/specificaties) of "BOTH",
        "gl_codes": [lijst van integers, bijv 4300],
        "search_terms": [lijst strings],
        "years": [lijst integers uit de beschikbare jaren],
        "reason": "Korte uitleg"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"source": "BOTH", "gl_codes": [], "search_terms": [], "years": available_years[-1:] if available_years else []}

def execute_csv_query(plan, df):
    """
    Stap 2: Filteren (Met super-robuste type-matching).
    """
    if df is None: return "", None
    
    subset = df.copy()
    debug_info = {"start_count": len(subset)}
    
    # 1. Filter op Jaar
    # We zorgen dat we alleen filteren als de planner zinnige jaren geeft
    req_years = plan.get("years", [])
    if req_years:
        subset = subset[subset['Finny_Year'].isin(req_years)]
    debug_info["after_year_filter"] = len(subset)
    debug_info["years_filtered"] = req_years
        
    # 2. Filter op GL Codes (De Hufterproof Matcher)
    req_codes = plan.get("gl_codes", [])
    if req_codes:
        # A. Zet de dataframe kolom om naar integers (4900.0 -> 4900)
        # We gebruiken pd.to_numeric om "4900.0" veilig naar getal te krijgen, dan int
        subset['TEMP_MATCH_CODE'] = pd.to_numeric(subset['Finny_GLCode'], errors='coerce').fillna(0).astype(int)
        
        # B. Zet de gevraagde codes om naar integers
        clean_req_codes = []
        for c in req_codes:
            try:
                clean_req_codes.append(int(float(c)))
            except:
                pass
        
        # C. Match
        subset = subset[subset['TEMP_MATCH_CODE'].isin(clean_req_codes)]
        debug_info["codes_searched"] = clean_req_codes
    
    debug_info["after_code_filter"] = len(subset)
        
    # 3. Filter op Zoektermen
    if plan.get("search_terms") and len(subset) > 0:
        terms = plan['search_terms']
        pattern = '|'.join(terms)
        mask = (
            subset['Description'].astype(str).str.contains(pattern, case=False, na=False) | 
            subset['AccountName'].astype(str).str.contains(pattern, case=False, na=False) |
            subset['Finny_GLDescription'].astype(str).str.contains(pattern, case=False, na=False)
        )
        # Alleen filteren als we nog geen GL codes hadden (anders is het te streng), 
        # tenzij expliciet als verfijning bedoeld. Laten we het toepassen als verfijning.
        subset = subset[mask]
        
    debug_info["final_count"] = len(subset)

    # 4. Resultaat Bouwen
    if len(subset) == 0:
        return "Geen transacties gevonden in CSV.", debug_info
        
    total = subset['AmountDC_num'].sum()
    
    if len(subset) > 40:
        summary = subset.groupby('Finny_GLDescription')['AmountDC_num'].sum().reset_index()
        summary['AmountDC_num'] = summary['AmountDC_num'].apply(lambda x: f"€ {x:,.2f}")
        txt = f"""
        **Resultaat uit Transacties (CSV):**
        - Aantal boekingen: {len(subset)}
        - **Totaal: € {total:,.2f}**
        
        **Per Categorie:**
        {summary.to_markdown(index=False)}
        """
        return txt, debug_info
    else:
        # Details
        disp_cols = ['EntryDate', 'AccountName', 'Description', 'AmountDC_num']
        # Check welke kolommen bestaan
        final_cols = [c for c in disp_cols if c in subset.columns]
        
        txt = f"""
        **Details uit Transacties (CSV):**
        - Totaal geselecteerd: **€ {total:,.2f}**
        
        """
        # Dataframe naar string zonder index
        txt += subset[final_cols].to_markdown(index=False)
        return txt, debug_info

# ==========================================
# 4. MAIN APP UI
# ==========================================
if check_password():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        st.error("API Key ontbreekt in secrets.")
        st.stop()

    # Sidebar
    with st.sidebar:
        if os.path.exists("finny_logo.jpg"):
            st.image("finny_logo.jpg", width=150)
        st.markdown("### 🏢 Finny Dashboard")
        
        data = load_data()
        
        if data["trans"] is not None:
            st.success(f"✅ CSV Geladen: {len(data['trans'])} regels")
            st.info(f"📅 Jaren in data: {data['years_available']}")
        else:
            st.error("❌ Geen CSV data")
            
        if st.button("Herstarten"): st.rerun()

    # Chat Interface
    st.title("🦁 Finny 5.4")
    st.markdown("Stel je vraag over de administratie.")

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Bijv: Wat zijn de autokosten in 2023?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            # 1. PLAN
            plan = get_finny_plan(client, prompt, data["syllabus"], data["years_available"])
            
            # 2. EXECUTE
            csv_context = ""
            debug_data = None
            
            if plan["source"] in ["CSV", "BOTH"]:
                csv_text, debug_data = execute_csv_query(plan, data["trans"])
                csv_context += csv_text
                
            if plan["source"] in ["PDF", "BOTH"]:
                csv_context += f"\n\nCONTEXT UIT PDF:\n{data['pdf_text']}"
            
            # 3. ANSWER
            try:
                system_msg = f"""
                Je bent Finny, de financieel assistent.
                Gebruik ONDERSTAANDE data om antwoord te geven.
                
                GEVONDEN DATA:
                {csv_context}
                
                INSTRUCTIES:
                - Als er CSV data is (tabellen/bedragen), gebruik die EXACT. Ga niet hallucineren.
                - Als er 'Geen transacties' staat, zeg dat dan eerlijk en suggereer een ander jaar of zoekterm.
                - Antwoord kort en zakelijk in Markdown.
                """
                
                completion = client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = completion.choices[0].message.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # 4. DEBUG EXPANDER (Voor jou, om te zien wat er misgaat)
                with st.expander("🔧 Finny's Denkproces (Debug Info)"):
                    st.write(f"**Strategie:** {plan.get('source')}")
                    st.write(f"**Gezochte Jaren:** {plan.get('years')}")
                    st.write(f"**Gezochte Codes:** {plan.get('gl_codes')}")
                    if debug_data:
                        st.write("--- Filter Stats ---")
                        st.write(f"Totaal regels data: {debug_data.get('start_count')}")
                        st.write(f"Na jaarfilter ({plan.get('years')}): {debug_data.get('after_year_filter')}")
                        st.write(f"Na codefilter ({plan.get('gl_codes')}): {debug_data.get('after_code_filter')}")
                        st.write("--- Data Voorbeeld ---")
                        st.dataframe(data["trans"].head(3))

            except Exception as e:
                st.error(f"Fout bij genereren antwoord: {e}")
