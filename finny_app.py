import streamlit as st
import pandas as pd
import os
import re
import glob
import json
from datetime import datetime
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import uuid
from pathlib import Path

# ==========================================
# 0. SETUP & CONFIG & CSS
# ==========================================

logo_files = glob.glob("finny_logo.png") + glob.glob("*.png") + glob.glob("*.jpg")
main_logo = "finny_logo.png" if os.path.exists("finny_logo.png") else (
    logo_files[0] if logo_files else "💰"
)

st.set_page_config(page_title="Finny Demo", page_icon=main_logo, layout="wide")
load_dotenv()

# --- CSS: VASTE FOOTER (ChatGPT STIJL) ---
# Dit zorgt voor de grijze balk onderin beeld.
st.markdown("""
    <style>
    /* Footer container vastzetten onderaan */
    .fixed-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        color: #6e6e6e;
        text-align: center;
        padding: 8px;
        font-size: 12px;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
    }
    
    /* Ruimte maken zodat chat niet achter de footer verdwijnt */
    .block-container {
        padding-bottom: 80px;
    }
    
    /* De input balk iets omhoog zodat hij niet achter de footer zit */
    .stChatInput {
        padding-bottom: 40px !important;
    }
    </style>
""", unsafe_allow_html=True)

AVAILABLE_YEARS = [2022, 2023, 2024]

# --- FALLBACK-SYNONIEMEN ---
FALLBACK_SYNONYMS = {
    "auto": ["auto", "autokosten", "bedrijfsauto", "lease", "brandstof", "tanken", "benzine", "laadpaal", "parkeer"],
    "telefoon": ["telefoon", "mobiel", "smartphone", "bellen", "internet", "wifi"],
    "personeel": ["personeel", "salaris", "lonen", "loonstrook", "werkgeverslasten", "pensioen"],
    "huisvesting": ["huisvesting", "huur", "gas", "water", "licht", "energie"],
    "verzekeringen": ["verzekering", "polis", "premie", "aov"],
    "bank": ["bankkosten", "rente", "bank"],
    "representatie": ["representatie", "lunch", "diner", "relatiegeschenk"],
    "reizen": ["reis", "hotel", "vliegticket", "taxi", "trein", "ov"],
}

YES_WORDS = {"ja", "ja.", "ja!", "graag", "zeker", "prima", "ok", "okay"}
NO_WORDS = {"nee", "nee.", "nee!", "liever niet", "hoeft niet", "laat maar"}

# ------------------------------------------
# 0A. LOGGING
# ------------------------------------------
BASE_DIR = Path(".")
LOG_DIR = BASE_DIR / "finny_logs"
LOG_DIR.mkdir(exist_ok=True)

CONSENT_LOG = LOG_DIR / "consent_log.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback_log.jsonl"
ESCALATION_LOG = LOG_DIR / "escalation_log.jsonl"

def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_event(log_file: Path, event_type: str, details: dict):
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": get_session_id(),
        "event_type": event_type,
        "details": details or {},
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

# --- STATE ---
if "active_years" not in st.session_state:
    st.session_state["active_years"] = AVAILABLE_YEARS
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_view" not in st.session_state:
    st.session_state.current_view = "intro"
if "client_profile" not in st.session_state:
    st.session_state.client_profile = {}
if "user_avatar_path" not in st.session_state:
    st.session_state.user_avatar_path = None
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None
if "legal_accepted" not in st.session_state:
    st.session_state.legal_accepted = False
if "profile_consent" not in st.session_state:
    st.session_state.profile_consent = False
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# ==========================================
# 1. AUTH & CONVERSATIONS
# ==========================================

def check_password() -> bool:
    if "password_correct" not in st.session_state:
        st.text_input(
            "Wachtwoord", type="password", key="pw",
            on_change=lambda: st.session_state.update({"password_correct": st.session_state.pw == "demo2025"}),
        )
        return False
    return bool(st.session_state["password_correct"])

def start_new_conversation() -> None:
    if st.session_state.messages:
        first_q = "Gesprek zonder titel"
        for m in st.session_state.messages:
            if m.get("role") == "user":
                content = str(m.get("content", ""))
                first_q = content[:60] + ("..." if len(content) > 60 else "")
                break
        new_conv = {
            "id": datetime.now().isoformat(),
            "title": first_q,
            "timestamp": datetime.now(),
            "messages": st.session_state.messages.copy(),
            "shared_with_accountant": False,
        }
        st.session_state.conversations.append(new_conv)

    st.session_state.messages = []
    st.session_state.feedback_given = False
    st.session_state["active_years"] = AVAILABLE_YEARS
    st.session_state.current_view = "chat"

# ==========================================
# 1B. LEGAL GATEWAY
# ==========================================

def show_legal_gateway():
    st.title("Welkom bij Finny")
    st.markdown("### Belangrijke informatie vooraf")
    
    st.info("Finny is een digitale AI-assistent. Om gebruik te maken van de demo, vragen wij u kennis te nemen van onderstaande voorwaarden.")
    
    with st.expander("Lees de volledige gebruiksvoorwaarden en disclaimer", expanded=False):
        st.markdown(
            """
            **1. Finny als AI-assistent**
            Finny is geen menselijke accountant en haar antwoorden vormen **géén professioneel financieel advies**.
            Raadpleeg bij belangrijke beslissingen altijd een professionele adviseur.

            **2. Gebruik op eigen risico**
            Buddy Workforce BV en uw accountant aanvaarden geen aansprakelijkheid voor fouten of schade.

            **3. Voorlopige cijfers & no-speculation**
            Cijfers zijn indicatief en niet gecontroleerd. Finny doet niet aan gissen.
            """
        )

    agreed = st.checkbox("Ik heb de disclaimer gelezen en ga akkoord.")
    st.divider()
    
    st.subheader("Toestemming voor opbouw klantprofiel")
    st.markdown("Finny kan een **persoonlijk klantprofiel** voor u opbouwen om u beter van dienst te zijn.")
    consent_profile = st.toggle("Ja, ik stem in met het analyseren van mijn gegevens.", value=False)
    
    if st.button("Start Finny", disabled=not agreed, type="primary"):
        st.session_state.legal_accepted = True
        st.session_state.profile_consent = consent_profile
        log_event(CONSENT_LOG, "legal_accepted", {"profile_consent": consent_profile})
        st.rerun()

# ==========================================
# 2. DATA LOAD
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    data = {"syn": None, "trans": None, "pdf_text": "", "company_context": "", "latest_year": max(AVAILABLE_YEARS)}
    
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", encoding="latin1")
            df.columns = df.columns.str.strip()
            if "Finny_Year" in df.columns:
                df["Year_Clean"] = df["Finny_Year"].astype(str).str.split(".").str[0].str.strip()
                valid = pd.to_numeric(df["Year_Clean"], errors="coerce").dropna()
                if not valid.empty:
                    data["latest_year"] = min(int(valid.max()), max(AVAILABLE_YEARS))
            
            for col in ["Description", "AccountName", "Finny_GLDescription", "Finny_GLCode", "AmountDC_num"]:
                if col not in df.columns: df[col] = ""
            
            df["UniversalSearch"] = df[["Description", "AccountName"]].astype(str).agg(" ".join, axis=1).str.lower()
            df["AmountDC_num"] = pd.to_numeric(df["AmountDC_num"], errors="coerce").fillna(0.0)
            data["trans"] = df
        except Exception: pass

    for pdf in glob.glob("*.pdf"):
        try:
            reader = PdfReader(pdf)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            data["pdf_text"] += f"\n--- PDF {pdf} ---\n{text}"
        except: continue
    return data

# ==========================================
# 3. ROUTER & LOGICA
# ==========================================
def classify_intent(client: OpenAI, question: str) -> dict:
    last = st.session_state.get("last_analysis") or {}
    system = (
        "Je bent een router. Bepaal intentie:\n"
        "- TOTAL_COST (totaalbedragen)\n"
        "- SPECIFIC_COST (specifieke post)\n"
        "- TREND (verloop over jaren)\n"
        "- DETAILS (transacties/lijstjes)\n"
        "- CHAT (algemeen)\n"
        "Geef JSON: { \"type\": \"...\", \"term\": \"...\" }"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": question}]
        )
        return json.loads(resp.choices[0].message.content)
    except:
        return {"type": "CHAT", "term": ""}

def extract_years(question: str, latest_year: int, intent_type: str) -> list[int]:
    years = sorted({int(y) for y in re.findall(r"\b(20[0-9]{2})\b", question) if int(y) in AVAILABLE_YEARS})
    if years: return years
    if intent_type == "TREND": 
        yrs = [y for y in AVAILABLE_YEARS if y <= latest_year]
        return yrs[-3:]
    return [latest_year]

def build_csv_query(data: dict, intent: dict, years: list[int]):
    df = data.get("trans")
    if df is None or df.empty: return None, None
    
    df = df[~df["Description"].astype(str).str.contains(r"(afsluiting|resultaat|balans)", case=False, na=False)]
    
    term = (intent.get("term") or "").lower()
    if intent["type"] in {"SPECIFIC_COST", "DETAILS"} and term:
        patterns = [re.escape(term)]
        for k, v in FALLBACK_SYNONYMS.items():
            if k in term: patterns.extend(v)
        pat = "|".join(patterns)
        df = df[df["UniversalSearch"].str.contains(pat, na=False)]
        
    str_years = [str(y) for y in years]
    df_curr = df[df["Year_Clean"].astype(str).isin(str_years)] if years else df
    
    if df_curr.empty: return None, None
    total = df_curr["AmountDC_num"].sum()
    
    lines = [f"Totaal geselecteerd: € {total:,.2f}"]
    
    if intent["type"] == "DETAILS" or (intent["type"] == "SPECIFIC_COST" and not df_curr.empty):
        top = df_curr.assign(Abs=df_curr["AmountDC_num"].abs()).sort_values("Abs", ascending=False).head(5)
        lines.append("\nTop 5 Transacties:")
        for _, row in top.iterrows():
            lines.append(f"- {row.get('Description')} (€ {row.get('AmountDC_num',0):,.2f})")
            
    return "\n".join(lines), total

def build_analysis(client, question, data):
    intent = classify_intent(client, question)
    years = extract_years(question, data["latest_year"], intent["type"])
    
    if intent["type"] == "CHAT":
        return {"context": data.get("pdf_text"), "source": "PDF", "intent": intent, "years": years}
    
    csv_txt, total = build_csv_query(data, intent, years)
    if csv_txt:
        return {"context": csv_txt, "source": "CSV", "intent": intent, "years": years}
    
    return {"context": data.get("pdf_text"), "source": "PDF", "intent": intent, "years": years}

# ==========================================
# 4. UI HELPERS (CLEAN - GEEN DISCLAIMERS HIER)
# ==========================================

ESCALATION_PHRASE = "Wilt u dat Finny uw vraag en relevante gegevens doorstuurt naar uw accountant?"

def render_escalation_if_needed(content: str, idx: int):
    # Alleen knoppen als de AI daarom vraagt. Verder NIKS.
    if ESCALATION_PHRASE in content:
        st.info("Finny stelt voor dit door te sturen naar je accountant.")
        c1, c2 = st.columns(2)
        if c1.button("Ja, stuur door", key=f"esc_y_{idx}"):
            log_event(ESCALATION_LOG, "escalation_approved", {"idx": idx})
            st.success("Verzonden!")
        if c2.button("Nee", key=f"esc_n_{idx}"):
            st.empty() 

# ==========================================
# 5. MAIN APP
# ==========================================
if check_password():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: st.stop()
    
    client = OpenAI(api_key=api_key)
    data = load_data()

    if not st.session_state.legal_accepted:
        show_legal_gateway()
        st.stop()

    # --- HIER IS DE VASTE FOOTER (GEMINI STYLE) ---
    # Dit staat buiten alle loops, altijd zichtbaar onderaan.
    st.markdown(
        '<div class="fixed-footer">'
        '⚠️ Finny kan fouten maken. Controleer belangrijke informatie altijd zelf. Geen professioneel advies.'
        '</div>', 
        unsafe_allow_html=True
    )

    # --- SIDEBAR ---
    with st.sidebar:
        st.image(main_logo, width=120)
        st.markdown("---")
        
        menu = {
            "intro": "👋 Kennismaking",
            "chat": "💬 Chat", 
            "history": "📂 Geschiedenis", 
            "share": "📩 Accountant"
        }
        
        opts = list(menu.keys())
        cur = st.session_state.current_view
        idx = opts.index(cur) if cur in opts else 0
        
        sel = st.radio("Menu", opts, index=idx, format_func=lambda x: menu[x])
        if sel != st.session_state.current_view:
            st.session_state.current_view = sel
            st.rerun()

        st.markdown("---")
        
        # Privacy & Info
        with st.expander("ℹ️ Privacy & Info"):
            st.markdown("Finny gebruikt AI. Uw data wordt verwerkt volgens onze [Privacyvoorwaarden](#).")
            st.toggle("Persoonlijk Profiel gebruiken", key="profile_consent", 
                      value=st.session_state.profile_consent, 
                      on_change=lambda: st.rerun())
        
        # ACTIEVE KNOP VOOR FEEDBACK - ALLEEN HIER!
        st.markdown("### Klaar met testen?")
        if st.button("Gesprek afronden", type="primary"):
            st.session_state.show_feedback_modal = True

    # --- FEEDBACK MODAL (ALLEEN NA KLIK) ---
    if st.session_state.get("show_feedback_modal") and not st.session_state.feedback_given:
        with st.container():
            st.markdown("---")
            st.markdown("### 👋 Hoe deed ik het vandaag?")
            c1, c2, c3 = st.columns([1,1,4])
            if c1.button("👍 Goed"):
                log_event(FEEDBACK_LOG, "session_rating", {"rating": "up"})
                st.session_state.feedback_given = True
                st.rerun()
            if c2.button("👎 Slecht"):
                log_event(FEEDBACK_LOG, "session_rating", {"rating": "down"})
                st.session_state.feedback_given = True
                st.rerun()

    if st.session_state.feedback_given:
        st.sidebar.success("Bedankt voor je feedback!")

    view = st.session_state.current_view

    # --- VIEW: INTRO ---
    if view == "intro":
        st.title("Jouw Finny Profiel")
        st.write("Help Finny om betere antwoorden te geven.")
        prof = st.session_state.client_profile
        with st.form("profiel_form"):
            col1, col2 = st.columns(2)
            with col1:
                kennis = st.slider("Financiële kennis", 1, 5, int(prof.get("finance_knowledge", 2)))
                focus_val = st.text_input("Jouw focus (bv. Groei, Kosten)", prof.get("focus", ""))
            with col2:
                detail = st.select_slider("Antwoord detail", ["Kort", "Normaal", "Uitgebreid"], value="Normaal")
                uploaded = st.file_uploader("Upload avatar/logo", type=["png","jpg"])
            
            if st.form_submit_button("Opslaan & naar Chat"):
                prof.update({"finance_knowledge": kennis, "focus": focus_val, "answer_detail": detail})
                if uploaded:
                    ext = os.path.splitext(uploaded.name)[1]
                    path = f"user_avatar{ext}"
                    with open(path, "wb") as f: f.write(uploaded.getbuffer())
                    st.session_state.user_avatar_path = path
                st.session_state.client_profile = prof
                st.session_state.current_view = "chat"
                st.rerun()

    # --- VIEW: CHAT ---
    elif view == "chat":
        st.title("Finny Demo")
        finny_avatar = main_logo if os.path.exists(main_logo) else "🤖"
        user_avatar = st.session_state.user_avatar_path if st.session_state.user_avatar_path else "👤"

        for idx, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"], avatar=finny_avatar if m["role"] == "assistant" else user_avatar):
                st.write(m["content"])
                if m["role"] == "assistant":
                    render_escalation_if_needed(m["content"], idx)

        prompt = st.chat_input("Vraag Finny iets over je cijfers...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user", avatar=user_avatar).write(prompt)
            
            with st.chat_message("assistant", avatar=finny_avatar):
                with st.spinner("Rekenmachine loopt..."):
                    pending = st.session_state.get("pending_followup")
                    question = prompt
                    if pending and prompt.lower() in YES_WORDS:
                        topic = pending.get("topic", "kosten")
                        years = pending.get("years", [])
                        question = f"Welke {topic} posten hadden de meeste impact in {years}?"
                        st.session_state.pending_followup = None
                    else:
                        st.session_state.pending_followup = None
                    
                    analysis = build_analysis(client, question, data)
                    st.session_state.last_analysis = analysis
                    
                    know = st.session_state.client_profile.get("finance_knowledge", 2)
                    tone = "Jip-en-Janneke" if know < 3 else "Zakelijk"
                    focus = st.session_state.client_profile.get("focus", "")
                    focus_instr = f"Hou rekening met de focus: {focus}." if focus else ""
                    
                    if analysis["intent"]["type"] == "TREND" and "winst" in question:
                        st.session_state.pending_followup = {"type": "TREND", "topic": "winst", "years": analysis["years"]}
                        followup_instr = "Vraag tot slot: 'Wil je dat ik laat zien welke kosten hier het meest aan bijdragen?'"
                    else:
                        followup_instr = ""

                    sys_prompt = (
                        f"Je bent Finny. Context: {analysis['context']}\n"
                        f"Tone: {tone}. {focus_instr} Antwoord kort. Geen verzinsels.\n"
                        f"Als je iets niet weet, zeg: 'Ik heb onvoldoende gegevens.'\n"
                        f"Als menselijke hulp nodig is, zeg exact: '{ESCALATION_PHRASE}'\n"
                        f"{followup_instr}"
                    )
                    
                    res = client.chat.completions.create(
                        model="gpt-4o-mini", messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": question}
                        ]
                    )
                    reply = res.choices[0].message.content
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    render_escalation_if_needed(reply, len(st.session_state.messages)-1)
    
    # --- VIEW: HISTORY ---
    elif view == "history":
        st.title("Geschiedenis")
        for c in st.session_state.conversations:
            with st.expander(f"{c['timestamp'].strftime('%d-%m %H:%M')} - {c['title']}"):
                for m in c["messages"]: st.write(f"**{m['role']}**: {m['content']}")

    # --- VIEW: SHARE ---
    elif view == "share":
        st.title("Delen met accountant")
        if not st.session_state.conversations:
            st.info("Er zijn nog geen gesprekken om te delen.")
        else:
            with st.form("share_form"):
                checks = {}
                for i, conv in enumerate(st.session_state.conversations):
                    title = conv.get("title", f"Gesprek {i+1}")
                    checked = conv.get("shared_with_accountant", False)
                    checks[i] = st.checkbox(title, value=checked, key=f"share_{i}")
                if st.form_submit_button("Opslaan"):
                    for i, v in checks.items():
                        st.session_state.conversations[i]["shared_with_accountant"] = bool(v)
                    st.success("Selectie bijgewerkt.")
                    st.rerun()
