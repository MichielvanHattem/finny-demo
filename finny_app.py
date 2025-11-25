import streamlit as st
import pandas as pd
import os
import json
import re
import glob
from datetime import datetime
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATIE & STATE
# ==========================================
st.set_page_config(page_title="Finny", layout="wide")
load_dotenv()

# HARDE EIS: Beschikbare jaren
AVAILABLE_YEARS = [2022, 2023, 2024]

# --- STATE MANAGEMENT INITIALISATIE ---
if "active_years" not in st.session_state:
    st.session_state["active_years"] = AVAILABLE_YEARS

if "messages" not in st.session_state: 
    st.session_state.messages = []

if "conversations" not in st.session_state:
    st.session_state.conversations = [] # Lijst voor logging

if "current_view" not in st.session_state:
    st.session_state.current_view = "chat" # chat, intro, history, share

if "client_profile" not in st.session_state:
    st.session_state.client_profile = {} # Leeg dict als default

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Wachtwoord", type="password", key="pw", on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}))
        return False
    return st.session_state["password_correct"]

# --- HULPFUNCTIES VOOR NAVIGATIE & LOGGING ---
def start_new_conversation():
    """Slaat huidig gesprek op en reset de chat."""
    if st.session_state.messages:
        # Bepaal titel op basis van eerste user message
        first_q = "Gesprek zonder titel"
        for m in st.session_state.messages:
            if m["role"] == "user":
                first_q = m["content"][:60] + ("..." if len(m["content"]) > 60 else "")
                break
        
        # Log het gesprek
        new_conv = {
            "id": datetime.now().isoformat(),
            "title": first_q,
            "timestamp": datetime.now(),
            "messages": st.session_state.messages.copy(),
            "shared_with_accountant": False
        }
        st.session_state.conversations.append(new_conv)
    
    # Reset
    st.session_state.messages = []
    st.session_state["active_years"] = AVAILABLE_YEARS
    # We blijven in de chat view na een nieuw gesprek
    st.session_state.current_view = "chat"

# ==========================================
# 2. DATA LADEN (Stabiele versie)
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    data = {"syn": None, "trans": None, "rgs": None, "pdf_text": "", "latest_year": 2024}

    def clean_code(val):
        return str(val).split('.')[0].strip()
    
    # A. TRANSACTIES
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", encoding="latin1") 
            
            if 'Finny_Year' in df.columns:
                df['Year_Clean'] = df['Finny_Year'].astype(str).str.split('.').str[0]
                valid_years = pd.to_numeric(df['Year_Clean'], errors='coerce').dropna()
                if not valid_years.empty:
                    data["latest_year"] = int(valid_years.max())
            
            # Universal Search kolom maken
            cols = ['Description', 'AccountName', 'Finny_GLDescription', 'Finny_GLCode']
            existing = [c for c in cols if c in df.columns]
            df['UniversalSearch'] = df[existing].astype(str).agg(' '.join, axis=1).str.lower()

            data["trans"] = df
        except Exception as e: st.error(f"Fout Transacties: {e}")

    # B. SYNONIEMEN
    if os.path.exists("Finny_Synonyms.csv"):
        try:
            df = pd.read_csv("Finny_Synonyms.csv", sep=";", dtype=str)
            if 'Finny_GLCode' in df.columns:
                df['Finny_GLCode'] = df['Finny_GLCode'].apply(clean_code)
            data["syn"] = df
        except: pass
    
    # C. RGS
    if os.path.exists("Finny_RGS.csv"):
        try:
            df = pd.read_csv("Finny_RGS.csv", sep=";", dtype=str)
            cols = ['RGS_Referentiecode', 'RGS_Omschrijving']
            if all(c in df.columns for c in cols):
                df['SearchBlob'] = df[cols].astype(str).agg(' '.join, axis=1).str.lower()
            data["rgs"] = df
        except: pass
    
    # D. PDF
    pdf_files = glob.glob("*.pdf")
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            data["pdf_text"] += f"\n--- {pdf} ---\n{text}"
        except: pass
        
    return data

# ==========================================
# 3. LOGICA & ROUTER
# ==========================================

def get_intent(client, question):
    """Router: Bepaalt Bron & Context."""
    context_years = st.session_state.get("active_years", AVAILABLE_YEARS)
    if not context_years: context_years = AVAILABLE_YEARS

    system_prompt = f"""
    Je bent de router van Finny.
    CONTEXT: Huidige focusjaren: {context_years}.
    
    TAAK:
    1. 'source': 'PDF' (winst/balans/omzet/totaalplaatje) of 'CSV' (specifieke kosten/details/leveranciers/trends).
    2. 'years': Welke jaren? ALS GEEN JAAR GENOEMD: Gebruik CONTEXT {AVAILABLE_YEARS}.
    3. 'keywords': Zoekwoorden (bv "telefoon", "auto", "vodafone").
    
    Output JSON.
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"}
        )
        intent = json.loads(res.choices[0].message.content)

        # ROUTER FIX: Harde override voor telefoon/communicatie
        q_lower = question.lower()
        TELEFOON_TERMS = ["telefoon", "telefoonkosten", "bellen", "mobiel", "vodafone", "ziggo", "communicatiekosten"]
        
        if any(t in q_lower for t in TELEFOON_TERMS):
            intent["source"] = "CSV"
            kws = intent.get("keywords") or []
            if not isinstance(kws, list): kws = []
            for t in TELEFOON_TERMS:
                if t in q_lower and t not in kws:
                    kws.append(t)
            intent["keywords"] = kws

        # Jaren verwerken
        if intent.get('years'):
            raw_years = intent['years']
            valid_years = [y for y in raw_years if int(y) in AVAILABLE_YEARS]
            if not valid_years:
                st.session_state["active_years"] = AVAILABLE_YEARS
                intent['years'] = AVAILABLE_YEARS
            else:
                st.session_state["active_years"] = valid_years
                intent['years'] = valid_years
        else:
            intent['years'] = context_years
            
        return intent
    except:
        return {"source": "PDF", "keywords": [], "years": context_years}

def analyze_csv_costs(data, intent):
    """Analyseert kosten uit CSV op basis van keywords/synoniemen."""
    if data["trans"] is None: return "Geen transacties."

    df = data["trans"].copy()
    keywords = intent.get("keywords", [])
    years = [str(y) for y in intent.get("years", AVAILABLE_YEARS)]

    # 1. Filter Jaren
    df = df[df['Year_Clean'].isin(years)]
    if df.empty: return f"Geen data gevonden voor de jaren {years}."
    
    # 2. Filter 'Echte' Kosten
    mask_tech = df['Description'].astype(str).str.contains(r'(resultaat|winst|balans|afsluiting)', case=False, na=False)
    df = df[~mask_tech]
    
    # 3. ZOEKEN & CATEGORISEREN
    found_categories = set()
    if keywords and data["syn"] is not None:
        for k in keywords:
            matches = data["syn"][data["syn"]['Synoniem'].str.contains(k.lower(), na=False)]
            if not matches.empty and 'Categorie' in matches.columns:
                found_categories.update(matches['Categorie'].unique().tolist())

    # 4. BEREKENINGEN
    # A. Specifieke Zoekopdracht
    df_specific = pd.DataFrame()
    if keywords:
        pattern = '|'.join([re.escape(k.lower()) for k in keywords])
        mask_spec = df['UniversalSearch'].str.contains(pattern, na=False)
        df_specific = df[mask_spec]

    # B. Categorie Totaal
    df_category = pd.DataFrame()
    if found_categories:
        cat_pattern = '|'.join([re.escape(c) for c in found_categories])
        mask_cat = df['Finny_GLDescription'].astype(str).str.contains(cat_pattern, case=False, na=False)
        df_category = df[mask_cat]
    else:
        if not keywords:
             df_category = df[df['Finny_GLCode'].str.startswith('4', na=False)]
    
    # 5. OUTPUT BOUWEN
    res = f"### ANALYSE ({', '.join(years)})\n"

    if not df_specific.empty:
        pivot_spec = df_specific.groupby('Year_Clean')['AmountDC_num'].sum().reset_index()
        pivot_spec.columns = ['Jaar', f'Specifiek "{", ".join(keywords)}"']
        res += pivot_spec.to_markdown(index=False, floatfmt=".2f") + "\n\n"
    elif keywords:
        res += f"Geen specifieke transacties gevonden voor '{keywords}'.\n\n"

    if not df_category.empty and found_categories:
        cat_names = ", ".join(found_categories)
        pivot_cat = df_category.groupby('Year_Clean')['AmountDC_num'].sum().reset_index()
        pivot_cat.columns = ['Jaar', f'Totaal Categorie: {cat_names}']
        res += f"**Context: Totale kosten in categorie '{cat_names}':**\n"
        res += pivot_cat.to_markdown(index=False, floatfmt=".2f") + "\n"

        res += f"\n*Grootste kostenposten binnen {cat_names}:*\n"
        top = df_category.groupby('Description')['AmountDC_num'].sum().sort_values(ascending=False).head(5).reset_index()
        res += top.to_markdown(index=False, floatfmt=".2f")

    return res

# ==========================================
# 4. UI HOOFDLOOP
# ==========================================
if check_password():
    api_key = os.getenv("OPENAI_API_KEY") 
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    
    if not api_key:
        st.error("Geen OpenAI API Key gevonden.")
        st.stop()

    client = OpenAI(api_key=api_key)
    data = load_data()
    
    # --- SIDEBAR & MENU ---
    with st.sidebar:
        # Logo
        logo_files = glob.glob("*.png") + glob.glob("*.jpg")
        if logo_files: st.image(logo_files[0], width=150)
        st.title("Finny")

        # Geheugen weergave
        if data["trans"] is not None:
            current_active = st.session_state.get("active_years", AVAILABLE_YEARS)
            st.caption(f"Geheugen: {current_active}")
            st.text(f"Beschikbaar: {AVAILABLE_YEARS}")
        
        st.markdown("---")
        
        # Nieuw gesprek knop
        if st.button("Nieuw Gesprek", use_container_width=True): 
            start_new_conversation()
            st.rerun()
            
        st.markdown("---")

        # MENU SELECTIE
        view_choice = st.radio(
            "Menu",
            ("chat", "intro", "history", "share"),
            format_func=lambda v: {
                "chat": "Chat",
                "intro": "Kennismaking",
                "history": "Eerdere gesprekken",
                "share": "Deel met accountant",
            }[v],
            index=["chat", "intro", "history", "share"].index(st.session_state.current_view)
        )
        
        if view_choice != st.session_state.current_view:
            st.session_state.current_view = view_choice
            st.rerun()

        # Shared Info
        shared_count = sum(1 for c in st.session_state.conversations if c.get("shared_with_accountant"))
        if shared_count > 0:
            st.markdown("---")
            st.info(f"Gemarkeerd voor accountant: {shared_count} gesprekken")


    # --- VIEW ROUTER ---
    view = st.session_state.current_view
    
    # --------------------------
    # VIEW: CHAT
    # --------------------------
    if view == "chat":
        st.title("Finny Demo")
        
        # Waarschuwing als gebruiker buiten bereik vraagt
        active = st.session_state.get("active_years", [])
        if any(y not in AVAILABLE_YEARS for y in active):
            st.warning(f"Let op: In deze demo zijn alleen de jaren {AVAILABLE_YEARS} beschikbaar.")

        for msg in st.session_state.messages: 
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Vraag Finny..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("..."):
                    # 1. Bepaal intent & Data
                    intent = get_intent(client, prompt)
                    context = ""
                    if intent['source'] == "PDF":
                        context = data["pdf_text"]
                        st.caption(f"Bron: Jaarrekening | Jaren: {intent['years']}")
                    else:
                        context = analyze_csv_costs(data, intent)
                        st.caption(f"Bron: Transacties | Focus: {intent['keywords']}")
                    
                    # 2. Profiel ophalen & integreren
                    profile = st.session_state.client_profile or {}
                    finance_level = profile.get("finance_knowledge", 3)
                    tax_level     = profile.get("tax_knowledge", 3)
                    book_level    = profile.get("bookkeeping_knowledge", 3)
                    risk_pref     = profile.get("risk_preference", 3)
                    focus_areas   = profile.get("focus_areas", "")
                    avoid_topics  = profile.get("avoid_topics", "")

                    profile_instructions = f"""
                    PROFIEL GEBRUIKER:
                    Kennis financiën: niveau {finance_level} (1=laag, 5=hoog)
                    Kennis belastingen: niveau {tax_level}
                    Kennis boekhouden: niveau {book_level}
                    Risicoprofiel: {risk_pref} (1=zeer risicomijdend, 5=risicobereid)
                    Focus: {focus_areas or '-'}
                    Vermijden: {avoid_topics or '-'}
                    
                    PAS JE UITLEG HIEROP AAN:
                    - Als kennisniveau laag is (<3), leg kernbegrippen kort uit in normale taal en vermijd vakjargon.
                    - Als kennisniveau hoog is (>=4), sla basisdefinities over en ga dieper in op nuances.
                    - Neem het risicoprofiel mee in advies (voorzichtig vs kansrijk).
                    - Ga niet in op 'Vermijden' onderwerpen tenzij gevraagd.
                    """.strip()

                    # 3. System Prompt Samenstellen
                    system_prompt_finny = f"""
                    Je bent Finny, een informele maar scherpe financiële assistent voor mkb-ondernemers.
                    STIJL:
                    - Spreek de gebruiker aan met 'je' en 'jij'.
                    - Geen brievenstijl, geen aanhef.
                    - Kort, duidelijk en eerlijk. Liever een "dat weet ik niet" dan verzonnen cijfers.
                    
                    INSTRUCTIES:
                    - Je krijgt onder DATA de relevante stukken uit de jaarrekening (PDF) of een analyse van de transacties (CSV).
                    - Gebruik alleen getallen die je in DATA ziet. Verzin geen bedragen.
                    - Gebruik de TABELLEN uit de context.
                    
                    {profile_instructions}
                    """
                    
                    messages_payload = [{"role": "system", "content": f"{system_prompt_finny}\n\nDATA:\n{context}"}]
                    for msg in st.session_state.messages[-5:]:
                        role = "user" if msg["role"] == "user" else "assistant"
                        messages_payload.append({"role": role, "content": msg["content"]})
                    
                    # 4. OpenAI Call
                    res = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages_payload
                    )
                    reply = res.choices[0].message.content
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

    # --------------------------
    # VIEW: KENNISMAKING (Intro)
    # --------------------------
    elif view == "intro":
        st.title("Kennismaking met Finny")
        st.write("Help Finny je beter te begrijpen door dit profiel in te vullen.")
        
        current_profile = st.session_state.client_profile or {}
        
        with st.form("intro_form"):
            st.subheader("Jouw kennisniveau")
            
            know_fin = st.slider("Kennis van financiën (winst, omzet)", 1, 5, current_profile.get("finance_knowledge", 2))
            know_tax = st.slider("Kennis van belastingen (BTW, VPB)", 1, 5, current_profile.get("tax_knowledge", 2))
            know_book = st.slider("Kennis van boekhouden", 1, 5, current_profile.get("bookkeeping_knowledge", 1))
            
            st.subheader("Voorkeuren")
            risk = st.select_slider(
                "Risico vs Zekerheid",
                options=[1, 2, 3, 4, 5],
                value=current_profile.get("risk_preference", 3),
                format_func=lambda x: {1: "1. Veiligheid eerst", 3: "3. Gemengd", 5: "5. Groeikansen pakken"}.get(x, str(x))
            )
            
            focus = st.text_area("Waar wil je dat Finny vooral bij helpt?", value=current_profile.get("focus_areas", ""))
            avoid = st.text_area("Onderwerpen die we over kunnen slaan?", value=current_profile.get("avoid_topics", ""))
            
            if st.form_submit_button("Profiel opslaan"):
                st.session_state.client_profile = {
                    "finance_knowledge": know_fin,
                    "tax_knowledge": know_tax,
                    "bookkeeping_knowledge": know_book,
                    "risk_preference": risk,
                    "focus_areas": focus,
                    "avoid_topics": avoid
                }
                st.success("Profiel opgeslagen!")
                st.rerun()
        
        if st.session_state.client_profile:
            p = st.session_state.client_profile
            st.markdown("### Je huidige profiel")
            st.write(f"- **Financiën:** niveau {p['finance_knowledge']}")
            st.write(f"- **Belastingen:** niveau {p['tax_knowledge']}")
            st.write(f"- **Boekhouden:** niveau {p['bookkeeping_knowledge']}")
            st.write(f"- **Risicoprofiel:** {p['risk_preference']}")
            st.write(f"- **Focus:** {p['focus_areas'] or '(nog niet ingevuld)'}")
            st.write(f"- **Overslaan:** {p['avoid_topics'] or '(geen)'}")

    # --------------------------
    # VIEW: EERDERE GESPREKKEN
    # --------------------------
    elif view == "history":
        st.title("Eerdere gesprekken")
        st.write("Hier vind je een overzicht van je sessies.")
        
        if not st.session_state.conversations:
            st.info("Nog geen gesprekken opgeslagen. Start een nieuw gesprek!")
        else:
            # Laatste eerst tonen
            for i, conv in enumerate(reversed(st.session_state.conversations)):
                with st.expander(f"{conv['timestamp'].strftime('%d-%m %H:%M')} - {conv['title']}"):
                    st.caption(f"Berichten: {len(conv['messages'])}")
                    for msg in conv['messages']:
                        role_label = "Gebruiker" if msg['role'] == "user" else "Finny"
                        st.markdown(f"**{role_label}:** {msg['content']}")

    # --------------------------
    # VIEW: DEEL MET ACCOUNTANT
    # --------------------------
    elif view == "share":
        st.title("Deel met accountant")
        st.write("Markeer de gesprekken die je wilt delen. Er wordt in deze demo nog niets daadwerkelijk verstuurd.")
        
        if not st.session_state.conversations:
            st.info("Geen gesprekken om te delen.")
        else:
            with st.form("share_form"):
                selection = {}
                for i, conv in enumerate(st.session_state.conversations):
                    label = f"{conv['timestamp'].strftime('%d-%m %H:%M')} - {conv['title']}"
                    checked = st.checkbox(
                        label,
                        value=conv.get("shared_with_accountant", False),
                        key=f"share_{i}"
                    )
                    selection[i] = checked
                    
                    with st.expander("Bekijk inhoud", expanded=False):
                        for msg in conv['messages']:
                            role = "Gebruiker" if msg['role'] == "user" else "Finny"
                            st.write(f"{role}: {msg['content'][:120]}")

                if st.form_submit_button("Bevestig selectie"):
                    for i, checked in selection.items():
                        st.session_state.conversations[i]['shared_with_accountant'] = checked
                    st.success("De gemarkeerde gesprekken zijn aangemerkt om te delen met je accountant.")
                    st.rerun()
