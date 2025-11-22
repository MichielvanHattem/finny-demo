import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

# Probeer imports die mogelijk falen veilig te laden
try:
    from openai import OpenAI
    from PyPDF2 import PdfReader
except ImportError as e:
    st.error(f"CRITISCHE FOUT: Ontbrekende bibliotheek. Zorg dat requirements.txt correct is. Detail: {e}")
    st.stop()

# ==========================================
# 1. SETUP & CONFIGURATIE
# ==========================================
st.set_page_config(page_title="Finny 8.0 | Hybride", page_icon="🦁", layout="wide")

# Veilige client initialisatie
client = None
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def render_logo():
    """Zoekt flexibel naar logo bestand."""
    logo_files = ["finny_logo.png", "finny_logo.jpg", "logo.png"]
    for l in logo_files:
        if os.path.exists(l):
            st.sidebar.image(l, width=140)
            return
    st.sidebar.markdown("### 🦁 FINNY")

# ==========================================
# 2. DATA MANAGER
# ==========================================
@st.cache_data
def load_data():
    data = {"csv": None, "pdf": "", "syllabus": "", "gl_map": None, "years": []}
    
    # A. CSV TRANSACTIES
    if os.path.exists("Finny_Transactions_Lite.csv"):
        try:
            df = pd.read_csv("Finny_Transactions_Lite.csv", sep=";", on_bad_lines='skip', low_memory=False)
            
            # Check op verplichte kolommen zoals uit analyse bleek
            required = ['AmountDC_num', 'Finny_GLCode']
            if not all(col in df.columns for col in required):
                st.error(f"CSV Fout: Mis kolommen {required}. Gevonden: {df.columns.tolist()}")
            else:
                # Normalisatie
                df['Finny_GLCode'] = pd.to_numeric(df['Finny_GLCode'], errors='coerce').fillna(0).astype(int).astype(str)
                if 'Finny_Year' in df.columns:
                    df['Finny_Year'] = pd.to_numeric(df['Finny_Year'], errors='coerce').fillna(0).astype(int)
                    data["years"] = sorted(df[df['Finny_Year'] > 2000]['Finny_Year'].unique().tolist())
                
                data["csv"] = df
        except Exception as e:
            st.error(f"Kon CSV niet laden: {e}")

    # B. PDF CONTENT
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for i, page in enumerate(reader.pages):
                if i < 20: text += page.extract_text() # Eerste 20 pagina's
            data["pdf"] += f"\n--- {pdf} ---\n{text}"
        except: pass

    # C. SYLLABUS & GL SCHEMA
    if os.path.exists("Finny_syllabus_v9_4_with_B3_B4_B5.txt"):
        with open("Finny_syllabus_v9_4_with_B3_B4_B5.txt", "r", encoding="utf-8") as f:
            data["syllabus"] = f.read()
            
    return data

# ==========================================
# 3. INTELLIGENTIE (ROUTER & LOGICA)
# ==========================================

def decide_strategy(question, syllabus_snippet):
    """
    De Hybride Router: Bepaalt of we naar PDF (Officieel) of CSV (Details) gaan.
    """
    if not client: return {"tool": "ERROR", "reason": "Geen API key"}
    
    prompt = f"""
    Je bent de architect van Finny. Analyseer de vraag.
    
    CONTEXT SYLLABUS: {syllabus_snippet[:1000]}
    
    VRAAG: "{question}"
    
    KIES GEREEDSCHAP:
    - "PDF": Voor vragen over totale Winst, Omzet, Balans, Jaarrekening tekst, Resultaatverdeling. (High level).
    - "CSV": Voor vragen over specifieke kostenposten (auto, huur), leveranciers, details, verloop over maanden. (Low level).
    - "BOTH": Als details én context nodig zijn.
    
    OUTPUT JSON:
    {{
        "tool": "PDF" | "CSV" | "BOTH",
        "reason": "Uitleg",
        "csv_search_terms": ["term1", "term2"] (alleen als CSV relevant is)
    }}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o", # We gebruiken het sterke model zoals geadviseerd
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        return {"tool": "PDF", "reason": "Fallback door fout", "csv_search_terms": []}

def get_gl_codes_for_query(question, syllabus, available_gl_codes):
    """
    Stap 1 van jouw proces: Vertaal Vraag -> GL Codes
    """
    prompt = f"""
    Vertaal de gebruikersvraag naar specifieke Grootboekrekeningnummers (GL Codes).
    Gebruik de syllabus voor synoniemen.
    
    VRAAG: {question}
    SYLLABUS: {syllabus[:3000]}
    
    Geef terug als JSON: {{ "gl_codes": ["4350", "4360", ...] }}
    Alleen nummers geven die logisch zijn voor deze kostensoort.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(res.choices[0].message.content).get("gl_codes", [])
    except:
        return []

def execute_csv_analysis(df, gl_codes, years=None):
    """
    Stap 2 & 3: Filter & Aggregeer
    """
    if df is None or not gl_codes: return "Geen data gevonden."
    
    # Filter op codes
    subset = df[df['Finny_GLCode'].isin([str(c) for c in gl_codes])]
    
    if len(subset) == 0: return "Geen transacties gevonden op deze GL codes."
    
    total = subset['AmountDC_num'].sum()
    
    # Groepeer per jaar
    per_year = subset.groupby('Finny_Year')['AmountDC_num'].sum().reset_index()
    per_year_str = per_year.to_string(index=False)
    
    # Top 5 details
    details = subset.sort_values('AmountDC_num').head(5)[['EntryDate', 'Description', 'AmountDC_num']]
    
    return f"""
    ANALYSE OP REKENINGEN: {gl_codes}
    --------------------------------
    TOTAAL BEDRAG: € {total:,.2f}
    
    PER JAAR:
    {per_year_str}
    
    VOORBEELD BOEKINGEN (Top 5):
    {details.to_string(index=False)}
    """

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
def main():
    # Authenticatie Check
    if "auth" not in st.session_state: st.session_state.auth = False
    
    if not st.session_state.auth:
        st.title("Finny 8.0 Login")
        pwd = st.text_input("Wachtwoord", type="password")
        if pwd == "demo2025":
            st.session_state.auth = True
            st.rerun()
        return

    # Check API key
    if not client:
        st.error("⚠️ Geen OpenAI API Key gevonden in st.secrets!")
        st.stop()

    # Sidebar
    with st.sidebar:
        render_logo()
        st.markdown("---")
        st.markdown("### 📡 Status")
        data = load_data()
        
        if data["csv"] is not None:
            st.success(f"Transacties: {len(data['csv'])}")
        else:
            st.error("Geen CSV data")
            
        if data["pdf"]:
            st.success("PDF Jaarrekening geladen")
            
        if st.button("Cache Wissen"):
            st.cache_data.clear()
            st.rerun()

    # Chat
    st.title("🦁 Finny 8.0")
    st.caption("Hybride Architectuur: PDF (Officieel) + CSV (Detail)")

    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: 
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if "meta" in m:
                with st.expander("Technische Analyse"):
                    st.json(m["meta"])

    if prompt := st.chat_input("Stel je vraag..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyseren en routeren..."):
                
                # STAP 1: ROUTER
                strategy = decide_strategy(prompt, data["syllabus"])
                context_text = ""
                debug_meta = {"strategy": strategy}
                
                # STAP 2: UITVOERING
                if strategy["tool"] in ["CSV", "BOTH"]:
                    # Vertaal vraag naar GL codes (De 'Pandas Agent' logica)
                    gl_codes = get_gl_codes_for_query(prompt, data["syllabus"], [])
                    debug_meta["gl_codes_found"] = gl_codes
                    
                    csv_result = execute_csv_analysis(data["csv"], gl_codes)
                    context_text += f"\n\n=== CSV DATA (Details) ===\n{csv_result}"
                
                if strategy["tool"] in ["PDF", "BOTH"]:
                    # Voeg PDF context toe
                    context_text += f"\n\n=== PDF DATA (Officieel) ===\n{data['pdf'][:15000]}" # Limit om tokens te sparen

                # STAP 3: ANTWOORD
                final_prompt = f"""
                Je bent Finny. Geef antwoord op de vraag.
                Gebruik de data hieronder.
                
                VRAAG: {prompt}
                
                DATA:
                {context_text}
                
                INSTRUCTIE:
                - Als de data uit de CSV komt, meld de specifieke rekeningnummers.
                - Als de data uit de PDF komt, vermeld "volgens de jaarrekening".
                - Wees zakelijk en concreet.
                """
                
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": final_prompt}],
                )
                
                ans = completion.choices[0].message.content
                st.write(ans)
                
                # Save history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ans, 
                    "meta": debug_meta
                })

if __name__ == "__main__":
    main()
