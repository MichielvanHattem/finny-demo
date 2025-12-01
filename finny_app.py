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
# 0. SETUP & CONFIG
# ==========================================

logo_files = glob.glob("finny_logo.png") + glob.glob("*.png") + glob.glob("*.jpg")
main_logo = "finny_logo.png" if os.path.exists("finny_logo.png") else (
    logo_files[0] if logo_files else "💰"
)

st.set_page_config(page_title="Finny Demo", page_icon=main_logo, layout="wide")
load_dotenv()

AVAILABLE_YEARS = [2022, 2023, 2024]

# --- FALLBACK-SYNONIEMEN (Optie D) ---
FALLBACK_SYNONYMS = {
    "auto": [
        "auto", "autokosten", "bedrijfsauto", "lease", "leaseauto", "brandstof",
        "tanken", "tankbeurt", "benzine", "diesel", "laadpaal", "laadkosten",
        "kilometervergoeding", "km-vergoeding", "parkeer", "wegenbelasting"
    ],
    "telefoon": [
        "telefoon", "telefonie", "telefoonkosten", "belkosten", "bellen",
        "mobiel", "smartphone", "sim", "bundel", "databundel",
        "internet", "wifi", "glasvezel", "microsoft 365", "teams"
    ],
    "personeel": [
        "personeel", "personeelskosten", "loonkosten", "salaris", "lonen",
        "loonstrook", "payroll", "werknemer", "vakantiegeld", "bonus",
        "provisie", "werkgeverslasten", "sociale lasten", "pensioen"
    ],
    "huisvesting": [
        "huisvesting", "huisvestingskosten", "kantoorpand", "bedrijfspand",
        "huur", "huur kantoor", "werkruimte", "bedrijfshuur",
        "gas", "water", "licht", "energie", "elektra", "nutsvoorzieningen"
    ],
    "verzekeringen": [
        "verzekering", "verzekeringen", "poliskosten", "premie",
        "aov", "arbeidsongeschiktheid", "bedrijfsaansprakelijkheid",
        "rechtsbijstand", "inventarisverzekering", "opstalverzekering",
        "autoverzekering", "wagenparkverzekering"
    ],
    "bank": [
        "bankkosten", "kosten bank", "abonnementskosten bank", "betaalpakket",
        "rente bank", "debetrente", "kredietrente", "rekening courant",
        "pintransactie", "creditcardkosten", "incassokosten"
    ],
    "representatie": [
        "representatie", "representatiekosten", "zakenlunch", "zakendiner",
        "borrel", "relatiegeschenk", "relatiegeschenken", "klantendiner",
        "netwerkborrel", "event", "beursbezoek"
    ],
    "reizen": [
        "reis", "reiskosten", "reis- en verblijfkosten", "hotel",
        "overnachting", "vliegticket", "vlucht", "taxi", "trein", "ov"
    ],
}

YES_WORDS = {
    "ja", "ja.", "ja!", "ja graag", "graag", "zeker", "prima",
    "is goed", "doe maar", "ok", "oke", "okay"
}
NO_WORDS = {
    "nee", "nee.", "nee!", "liever niet", "hoeft niet", "laat maar",
    "niet nodig", "nee hoor"
}

# ------------------------------------------
# 0A. LOGGING / AVG-VRIENDELIJKE STORAGE
# ------------------------------------------
BASE_DIR = Path(".")
LOG_DIR = BASE_DIR / "finny_logs"
LOG_DIR.mkdir(exist_ok=True)

CONSENT_LOG = LOG_DIR / "consent_log.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback_log.jsonl"
ESCALATION_LOG = LOG_DIR / "escalation_log.jsonl"


def get_session_id() -> str:
    """Genereer een anonieme sessie-ID (geen naam, alleen UUID)."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def log_event(log_file: Path, event_type: str, details: dict):
    """Schrijf AVG-proof logregel (geen naam, alleen sessie, timestamp & type)."""
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
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []
# juridische vlaggen
if "legal_accepted" not in st.session_state:
    st.session_state.legal_accepted = False
if "profile_consent" not in st.session_state:
    st.session_state.profile_consent = False
if "profile_consent_timestamp" not in st.session_state:
    st.session_state.profile_consent_timestamp = None
if "pending_escalation_message_id" not in st.session_state:
    st.session_state.pending_escalation_message_id = None
# escalatie + sessie-feedback
if "escalation_consent" not in st.session_state:
    st.session_state.escalation_consent = None  # None / True / False
if "escalated_message_ids" not in st.session_state:
    st.session_state.escalated_message_ids = []
if "session_feedback_given" not in st.session_state:
    st.session_state.session_feedback_given = False
if "show_conv_feedback_input" not in st.session_state:
    st.session_state.show_conv_feedback_input = False
if "conversation_finished" not in st.session_state:
    st.session_state.conversation_finished = False

# ==========================================
# 1. AUTH & CONVERSATIONS
# ==========================================

def check_password() -> bool:
    if "password_correct" not in st.session_state:
        st.text_input(
            "Wachtwoord",
            type="password",
            key="pw",
            on_change=lambda: st.session_state.update(
                {"password_correct": st.session_state.pw == "demo2025"}
            ),
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
    st.session_state["active_years"] = AVAILABLE_YEARS
    st.session_state.current_view = "chat"
    st.session_state.session_feedback_given = False
    st.session_state.show_conv_feedback_input = False
    st.session_state.conversation_finished = False
    st.session_state.escalation_consent = None
    st.session_state.escalated_message_ids = []


# ==========================================
# 1B. LEGAL GATEWAY (DISCLAIMER + PROFILERING)
# ==========================================

def show_legal_gateway():
    st.title("Welkom bij Finny")

    with st.expander("Lees de gebruiksvoorwaarden en disclaimer", expanded=True):
        st.markdown(
            """
**Finny als AI-assistent**

Finny is een digitale AI-assistent die u helpt inzicht te krijgen in uw financiële cijfers. 
Finny is geen menselijke accountant en haar antwoorden vormen géén professioneel financieel advies 
of accountantsverklaring. U mag de informatie van Finny niet beschouwen als een definitief oordeel 
of advies van een bevoegd accountant; raadpleeg bij belangrijke beslissingen altijd een 
professionele adviseur.

**Gebruik op eigen risico & aansprakelijkheid**

Het gebruik van Finny is volledig voor eigen risico. Hoewel Finny met zorg is ontwikkeld, 
kunnen antwoorden onjuist of onvolledig zijn. Buddy Workforce BV (ontwikkelaar van Finny) en 
uw accountant aanvaarden geen enkele aansprakelijkheid voor fouten, schade of beslissingen die 
voortvloeien uit het gebruik van Finny’s antwoorden. U blijft zelf verantwoordelijk voor de keuzes 
die u maakt op basis van de door Finny gegeven informatie.

**Voorlopige cijfers & no-speculation**

Antwoorden of rapportages van Finny kunnen voorlopige financiële cijfers bevatten. 
Deze voorlopige cijfers zijn enkel indicatief, niet gecontroleerd door een accountant en kunnen nog wijzigen. 
Finny doet niet aan gissen of speculatie: als de benodigde betrouwbare gegevens ontbreken, zal Finny geen 
antwoord geven in plaats van te raden. Dit betekent dat u soms het bericht krijgt dat er geen betrouwbaar 
antwoord beschikbaar is, wanneer Finny onvoldoende zeker is van de informatie.
            """
        )

    agreed = st.checkbox("Ik heb de disclaimer gelezen en ga akkoord.")

    st.divider()
    st.subheader("Toestemming voor opbouw klantprofiel")

    st.info(
        "Finny kan een persoonlijk klantprofiel voor u opbouwen om u beter van dienst te zijn. "
        "Daarvoor analyseert Finny uw relevante financiële gegevens en eerdere interacties, zodat "
        "antwoorden beter op uw situatie aansluiten. U beslist zelf of u dit wilt. "
        "U kunt uw keuze later altijd wijzigen via de instellingen."
    )

    consent_profile = st.checkbox(
        "Ik stem in met het analyseren van mijn gegevens voor een persoonlijk klantprofiel."
    )

    start = st.button("Start Finny", disabled=not agreed, type="primary")

    if start:
        st.session_state.legal_accepted = True
        st.session_state.profile_consent = consent_profile
        if consent_profile:
            st.session_state.profile_consent_timestamp = datetime.utcnow().isoformat()
            log_event(
                CONSENT_LOG,
                "profile_consent_granted",
                {"consent": True},
            )
        else:
            log_event(
                CONSENT_LOG,
                "profile_consent_denied",
                {"consent": False},
            )

        log_event(
            CONSENT_LOG,
            "legal_disclaimer_accepted",
            {"agreed": True},
        )

        st.rerun()


# ==========================================
# 2. DATA LOAD
# ==========================================

@st.cache_data(ttl=3600)
def load_data():
    data = {
        "syn": None,
        "trans": None,
        "pdf_text": "",
        "company_context": "",
        "latest_year": max(AVAILABLE_YEARS),
    }

    # Transacties (CSV)
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", encoding="latin1")
            df.columns = df.columns.str.strip()

            if "Finny_Year" in df.columns:
                df["Year_Clean"] = (
                    df["Finny_Year"].astype(str).str.split(".").str[0].str.strip()
                )
                valid_years = pd.to_numeric(df["Year_Clean"], errors="coerce").dropna()
                if not valid_years.empty:
                    latest = int(valid_years.max())
                    data["latest_year"] = min(latest, max(AVAILABLE_YEARS))
                    st.session_state["active_years"] = sorted(
                        y for y in AVAILABLE_YEARS if y <= data["latest_year"]
                    )

            for col in [
                "Description",
                "AccountName",
                "Finny_GLDescription",
                "Finny_GLCode",
                "AmountDC_num",
            ]:
                if col not in df.columns:
                    df[col] = ""

            df["UniversalSearch"] = (
                df[["Description", "AccountName", "Finny_GLDescription", "Finny_GLCode"]]
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )
            df["AmountDC_num"] = pd.to_numeric(
                df["AmountDC_num"], errors="coerce"
            ).fillna(0.0)

            data["trans"] = df
        except Exception as e:
            st.error(f"Fout bij laden Finny_Transactions.csv: {e}")

    # Synoniemen (extern beheerd)
    if os.path.exists("Finny_Synonyms.csv"):
        try:
            syn = pd.read_csv("Finny_Synonyms.csv", sep=";", dtype=str)
            if "Synoniem" in syn.columns:
                syn["Synoniem_Clean"] = (
                    syn["Synoniem"].astype(str).str.lower().str.strip()
                )
            data["syn"] = syn
        except Exception:
            data["syn"] = None

    # Jaarrekening PDFs
    pdf_files = glob.glob("*.pdf")
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text_parts = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                text_parts.append(t)
            all_text = "\n".join(text_parts)
            data["pdf_text"] += f"\n--- JAARREKENING {pdf} ---\n{all_text}"
        except Exception:
            continue

    # Bedrijfsprofiel TXT
    txt_files = glob.glob("*profiel*.txt") + glob.glob("*profile*.txt")
    for txt in txt_files:
        try:
            with open(txt, "r", encoding="utf-8") as f:
                data["company_context"] += f"\n--- PROFIEL ({txt}) ---\n{f.read()}\n"
        except Exception:
            try:
                with open(txt, "r", encoding="latin1") as f:
                    data["company_context"] += f"\n--- PROFIEL ({txt}) ---\n{f.read()}\n"
            except Exception:
                continue

    return data


# ==========================================
# 3. ROUTER & CSV-LOGICA (SANDWICH)
# ==========================================

def classify_intent(client: OpenAI, question: str) -> dict:
    """
    Context-gevoelige router.
    """
    last = st.session_state.get("last_analysis") or {}
    last_intent = last.get("intent") or {}
    last_type = last_intent.get("type")
    last_term = last_intent.get("term")
    last_years = last.get("years")

    last_summary = {
        "last_type": last_type,
        "last_term": last_term,
        "last_years": last_years,
    }

    system = (
        "Je bent een router voor de Finny demo.\n"
        "Je krijgt een nieuwe vraag van een ondernemer én een korte samenvatting "
        "van de vorige analyse.\n\n"
        "Je taak:\n"
        "- Bepaal of de nieuwe vraag een 'NEW' vraag is of een 'FOLLOW_UP' op de vorige.\n"
        "- Bepaal het soort analyse in 'analysis_type':\n"
        "    - 'TOTAL_COST'   → totaal van kosten/uitgaven/resultaat in een periode\n"
        "    - 'SPECIFIC_COST'→ specifieke post (telefoon, auto, personeel, huur, etc.)\n"
        "    - 'TREND'        → verloop/vergelijking over jaren\n"
        "    - 'DETAILS'      → drill-down / individuele transacties / specificatie\n"
        "    - 'CHAT'         → algemene uitleg of praat zonder directe cijferanalyse\n"
        "- Bepaal 'term' als het onderwerp van de vraag in gewone Nederlandse woorden.\n\n"
        "Geef ALLEEN een JSON-object terug met:\n"
        "{\n"
        '  \"relation\": \"NEW\" | \"FOLLOW_UP\",\n'
        '  \"analysis_type\": \"TOTAL_COST\" | \"SPECIFIC_COST\" | \"TREND\" | \"DETAILS\" | \"CHAT\",\n'
        '  \"term\": \"<onderwerp in gewone woorden>\"\n'
        "}\n"
    )

    user_payload = {
        "question": question,
        "last_summary": last_summary,
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
        )
        data_resp = json.loads(resp.choices[0].message.content)

        analysis_type = (data_resp.get("analysis_type") or data_resp.get("type") or "CHAT").upper()
        term = (data_resp.get("term") or "").strip()
        relation = (data_resp.get("relation") or "NEW").upper()

        if analysis_type not in {"TOTAL_COST", "SPECIFIC_COST", "TREND", "DETAILS", "CHAT"}:
            analysis_type = "CHAT"
        if relation not in {"NEW", "FOLLOW_UP"}:
            relation = "NEW"

        return {"type": analysis_type, "term": term, "relation": relation}
    except Exception:
        return {"type": "CHAT", "term": "", "relation": "NEW"}


def extract_years(question: str, latest_year: int, intent_type: str) -> list[int]:
    """Jaartallen pas ná intent detecteren."""
    years = sorted(
        {int(y) for y in re.findall(r"\b(20[0-9]{2})\b", question) if int(y) in AVAILABLE_YEARS}
    )
    if years:
        return years

    if intent_type in {"TOTAL_COST", "SPECIFIC_COST", "DETAILS"}:
        return [latest_year]
    if intent_type == "TREND":
        yrs = [y for y in AVAILABLE_YEARS if y <= latest_year]
        return yrs[-3:] if len(yrs) >= 3 else yrs
    return []


def _extend_patterns_with_fallback(term: str, patterns: list[str]) -> list[str]:
    term_l = term.lower()
    extra = []
    for key, syns in FALLBACK_SYNONYMS.items():
        if key in term_l or term_l in key:
            extra.extend(syns)
    if extra:
        patterns.extend(extra)
    return patterns


def classify_followup_answer(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return "OTHER"
    word_count = len(t.split())
    if any(w in t for w in YES_WORDS) and word_count <= 6:
        return "CONFIRM"
    if any(w in t for w in NO_WORDS) and word_count <= 6:
        return "DECLINE"
    return "OTHER"


def extract_optional_topic_from_text(text: str) -> str | None:
    t = text.lower()
    for key in FALLBACK_SYNONYMS.keys():
        if key in t:
            return key
    return None


def detect_simple_followup(text: str, last_analysis: dict | None, data: dict) -> str | None:
    """
    Herken simpele 'waarom / hoe komt dat'-vervolgvragen -> EXPLAIN_ONLY,
    MAAR niet als de vraag duidelijk over details / transacties / maanden gaat.
    """
    if not last_analysis:
        return None

    t = text.strip().lower()
    if not t:
        return None

    # Lange vragen: beschouw als nieuwe inhoudelijke vraag
    if len(t.split()) > 20:
        return None

    # Als er expliciet een jaar in staat → opnieuw rekenen, geen explain-only
    if re.search(r"\b20[0-9]{2}\b", t):
        return None

    # Vragen over transacties / per maand / overzicht moeten juist de data in
    detail_stop_words = ["transactie", "transacties", "per maand", "maand", "maandelijks"]
    if any(w in t for w in detail_stop_words):
        return None

    # Triggerwoorden voor een uitleg-vraag
    trigger_patterns = [
        r"\bwaarom\b",
        r"hoe komt dat",
        r"hoe kan dat",
        r"hoezo",
        r"waar komt dat door",
        r"waar ligt dat aan",
    ]
    if not any(re.search(p, t) for p in trigger_patterns):
        return None

    original_question = text.strip()

    synthetic_question = (
        "[FOLLOWUP_EXPLAIN] "
        "De ondernemer stelt nu de vervolg-vraag: '"
        + original_question
        + "'. "
        "Gebruik ALLEEN de bedragen en verhoudingen die eerder in dit gesprek zijn genoemd "
        "(bijvoorbeeld winst, autokosten en brandstofkosten) en geef in gewone taal een korte uitleg "
        "waarom die cijfers zo kunnen uitpakken. "
        "Ga NIET opnieuw rekenen op basis van CSV- of PDF-data, maar leg het principe uit "
        "(bijvoorbeeld: een totaalpost bestaat uit meerdere deelposten, dus een stijging in brandstof "
        "kan worden gecompenseerd door daling in andere autokosten zoals lease, onderhoud of verzekering)."
    )

    return synthetic_question


def build_csv_query(data: dict, intent: dict, years: list[int]) -> tuple[str | None, float | None]:
    """
    CSV-aggregatie + DETAILS-drill-down.
    """
    base_df = data.get("trans")
    syn = data.get("syn")
    if base_df is None or base_df.empty:
        return None, None

    df = base_df.copy()

    if "Description" in df.columns:
        df = df[
            ~df["Description"]
            .astype(str)
            .str.contains(
                r"(afsluiting|resultaat boekjaar|winstbestemming|memoriaal|balans|privé)",
                case=False,
                na=False,
            )
        ]

    if "Year_Clean" not in df.columns and "Finny_Year" in df.columns:
        df["Year_Clean"] = (
            df["Finny_Year"].astype(str).str.split(".").str[0].str.strip()
        )

    df["AmountDC_num"] = pd.to_numeric(df.get("AmountDC_num", 0), errors="coerce").fillna(0.0)

    intent_type = intent["type"]
    term = (intent["term"] or "").lower().strip()

    if intent_type in {"SPECIFIC_COST", "DETAILS"} and term:
        patterns: list[str] = [re.escape(term)]
        used_csv_synonyms = False

        if syn is not None and "Synoniem_Clean" in syn.columns:
            mask = syn["Synoniem_Clean"].astype(str).str.contains(term, case=False, na=False)
            if mask.any():
                used_csv_synonyms = True
                syn_terms = syn.loc[mask, "Synoniem_Clean"].dropna().unique().tolist()
                patterns.extend(re.escape(t) for t in syn_terms)
                for col in ["Categorie", "Finny_GLDescription", "Finny_GLCode"]:
                    if col in syn.columns:
                        patterns.extend(
                            re.escape(str(v).lower())
                            for v in syn.loc[mask, col].dropna().unique().tolist()
                        )

        if not used_csv_synonyms:
            patterns = _extend_patterns_with_fallback(term, patterns)

        pat = "|".join(sorted(set(patterns)))
        if "UniversalSearch" in df.columns and pat:
            df = df[df["UniversalSearch"].astype(str).str.contains(pat, na=False)]

    if df.empty:
        return None, None

    available_years_in_df = sorted(
        {int(y) for y in pd.to_numeric(df.get("Year_Clean", []), errors="coerce").dropna()}
    )

    str_years = [str(y) for y in years] if years else None
    if str_years and "Year_Clean" in df.columns:
        df_current_range = df[df["Year_Clean"].astype(str).isin(str_years)]
    else:
        df_current_range = df

    if df_current_range.empty:
        return None, None

    total = float(df_current_range["AmountDC_num"].sum())
    if abs(total) < 1e-6:
        return None, None

    yrs_label = ", ".join(str(y) for y in years) if years else "alle jaren"
    lines: list[str] = []

    def trend_text(curr: float, prev: float, label: str) -> str:
        if abs(prev) < 1e-6:
            return f"{label} in het vorige jaar was nul of verwaarloosbaar; een percentage is daardoor niet zinvol."
        diff = curr - prev
        pct = (diff / abs(prev)) * 100.0
        richting = "gestegen" if diff > 0 else ("gedaald" if diff < 0 else "ongeveer gelijk gebleven")
        return (
            f"{label} is {richting} met € {diff:,.0f} "
            f"({pct:+.1f}% ten opzichte van het jaar ervoor)."
        )

    if intent_type == "DETAILS":
        df_sorted = df_current_range.copy()
        df_sorted["AbsAmount"] = df_sorted["AmountDC_num"].abs()
        top_trans = df_sorted.sort_values(by="AbsAmount", ascending=False).head(10)

        lines.append(
            f"### Top transacties voor '{term or 'geselecteerde kosten'}' ({yrs_label})"
        )
        for _, row in top_trans.iterrows():
            desc = str(row.get("Description", "") or "").strip() or "Onbekend"
            acc = str(row.get("AccountName", "") or "").strip()
            date = str(row.get("Date", "") or "").strip()
            amt = float(row.get("AmountDC_num", 0.0))

            label_parts = []
            if date:
                label_parts.append(date)
            if acc:
                label_parts.append(acc)
            label = " | ".join(label_parts) if label_parts else "Transactie"

            lines.append(f"- {label}: {desc} → € {amt:,.2f}")

        return "\n".join(lines), total

    if intent_type in {"TOTAL_COST", "SPECIFIC_COST"}:
        if years and len(years) == 1 and "Year_Clean" in df.columns:
            cur_year = years[0]
            df_cur = df_current_range[df_current_range["Year_Clean"].astype(str) == str(cur_year)]
            total_cur = float(df_cur["AmountDC_num"].sum())

            prev_year = cur_year - 1 if (cur_year - 1) in available_years_in_df else None
            prev_line = ""
            if prev_year is not None:
                df_prev = df[df["Year_Clean"].astype(str) == str(prev_year)]
                total_prev = float(df_prev["AmountDC_num"].sum())
                prev_line = trend_text(total_cur, total_prev, f"Ten opzichte van {prev_year}")
            else:
                total_prev = None

            label = "totale kosten" if intent_type == "TOTAL_COST" else (term or "geselecteerde kosten")

            lines.append(f"### {label.capitalize()} in {cur_year}")
            lines.append(f"Totaal: € {total_cur:,.2f}")

            if total_prev is not None:
                lines.append(prev_line)

            if "Description" in df_cur.columns:
                top = (
                    df_cur.groupby("Description")["AmountDC_num"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                    .reset_index()
                )
                if not top.empty:
                    lines.append("")
                    lines.append("Top 5 posten in dit jaar:")
                    lines.append(top.to_markdown(index=False, floatfmt=".2f"))

            return "\n".join(lines), total_cur

        label = "totale kosten" if intent_type == "TOTAL_COST" else (term or "geselecteerde kosten")
        lines.append(f"### {label.capitalize()} ({yrs_label})")
        lines.append(f"Totaal: € {total:,.2f}")

        if "Year_Clean" in df_current_range.columns:
            per_year = (
                df_current_range.groupby("Year_Clean")["AmountDC_num"]
                .sum()
                .reset_index()
                .sort_values("Year_Clean")
            )
            if len(per_year) >= 2:
                first_row = per_year.iloc[0]
                last_row = per_year.iloc[-1]
                try:
                    y1 = int(first_row["Year_Clean"])
                    y2 = int(last_row["Year_Clean"])
                    t1 = float(first_row["AmountDC_num"])
                    t2 = float(last_row["AmountDC_num"])
                    lines.append("")
                    lines.append(trend_text(t2, t1, f"Tussen {y1} en {y2}"))
                except Exception:
                    pass

            lines.append("")
            lines.append("Overzicht per jaar:")
            lines.append(per_year.to_markdown(index=False, floatfmt=".2f"))

        return "\n".join(lines), total

    if intent_type == "TREND":
        lines.append(f"### Verloop ({yrs_label}) – samenvatting uit transacties")
        if "Year_Clean" in df_current_range.columns:
            per_year = (
                df_current_range.groupby("Year_Clean")["AmountDC_num"]
                .sum()
                .reset_index()
                .sort_values("Year_Clean")
            )
            lines.append("")
            lines.append(per_year.to_markdown(index=False, floatfmt=".2f"))

            if len(per_year) >= 2:
                first_row = per_year.iloc[0]
                last_row = per_year.iloc[-1]
                try:
                    y1 = int(first_row["Year_Clean"])
                    y2 = int(last_row["Year_Clean"])
                    t1 = float(first_row["AmountDC_num"])
                    t2 = float(last_row["AmountDC_num"])
                    lines.append("")
                    lines.append(trend_text(t2, t1, f"Over de periode {y1}–{y2}"))
                except Exception:
                    pass
        else:
            lines.append(f"Totaal alle jaren samen: € {total:,.2f}")

        return "\n".join(lines), total

    lines.append(f"### Cijfers uit transacties ({yrs_label})")
    lines.append(f"Totaal: € {total:,.2f}")
    return "\n".join(lines), total


def build_analysis(client: OpenAI, question: str, data: dict) -> dict:
    intent = classify_intent(client, question)
    intent_type = intent["type"]
    latest_year = data.get("latest_year", max(AVAILABLE_YEARS))

    years = extract_years(question, latest_year, intent_type)

    if intent_type == "CHAT":
        context = data.get("pdf_text") or ""
        source = "PDF"
        total = None
    else:
        csv_context, total = build_csv_query(data, intent, years)
        if csv_context is not None:
            context = csv_context
            source = "CSV"
        else:
            context = data.get("pdf_text") or ""
            source = "PDF"

    return {
        "intent": intent,
        "years": years,
        "source": source,
        "context": context,
        "total": total,
    }


def build_conversation_snippet(max_turns: int = 2) -> str:
    log = st.session_state.get("conversation_log", [])
    if not log:
        return ""

    recent = log[-max_turns:]
    lines: list[str] = []
    for turn in recent:
        q = (turn.get("user_question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if not q and not a:
            continue
        lines.append(f"- Vraag: {q}\n  Antwoord: {a}")

    return "\n".join(lines)


# ==========================================
# 3B. UI HULPFUNCTIE: ESCALATIE
# ==========================================

ESCALATION_PHRASE = "Wilt u dat Finny uw vraag en relevante gegevens doorstuurt naar uw accountant?"


def render_assistant_extras(content: str, idx: int):
    """
    Toon alleen de escalatie-opties onder een Finny-antwoord.
    Geen disclaimer en geen thumbs per antwoord.
    """
    if ESCALATION_PHRASE not in content:
        return

    consent = st.session_state.get("escalation_consent", None)

    # 1. Nog geen keuze gemaakt in deze sessie -> vraag eenmalig toestemming
    if consent is None:
        st.info(
            "Finny kan deze vraag met uw toestemming doorsturen naar uw accountant, "
            "inclusief relevante financiële context."
        )
        c1, c2 = st.columns(2)
        with c1:
            yes = st.button(
                "Ja, doorsturen naar accountant",
                key=f"escalate_yes_{idx}",
            )
        with c2:
            no = st.button(
                "Nee, liever niet",
                key=f"escalate_no_{idx}",
            )

        if yes:
            st.session_state.escalation_consent = True
            log_event(
                ESCALATION_LOG,
                "escalation_consent_granted",
                {},
            )
            if idx not in st.session_state.escalated_message_ids:
                st.session_state.escalated_message_ids.append(idx)
                log_event(
                    ESCALATION_LOG,
                    "escalation_approved",
                    {"message_index": idx, "snippet": content[:300]},
                )
            st.success(
                "Uw vraag is door Finny gemarkeerd om naar uw accountant te worden doorgestuurd."
            )

        if no:
            st.session_state.escalation_consent = False
            log_event(
                ESCALATION_LOG,
                "escalation_consent_declined",
                {},
            )
            st.info(
                "Begrepen, vragen in dit gesprek worden niet door Finny naar uw accountant doorgestuurd."
            )

    # 2. Toestemming al gegeven: automatisch markeren, geen nieuwe pop-up
    elif consent is True:
        st.info(
            "Deze vraag wordt volgens uw eerdere toestemming door Finny gemarkeerd "
            "om naar uw accountant te worden doorgestuurd."
        )
        if idx not in st.session_state.escalated_message_ids:
            st.session_state.escalated_message_ids.append(idx)
            log_event(
                ESCALATION_LOG,
                "escalation_auto_approved",
                {"message_index": idx, "snippet": content[:300]},
            )

    # 3. Toestemming eerder geweigerd: alleen uitleg tonen
    else:
        st.info(
            "U heeft eerder aangegeven dat u vragen in dit gesprek niet wilt doorsturen "
            "naar uw accountant."
        )


# ==========================================
# 4. MAIN UI
# ==========================================

if check_password():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Geen OPENAI_API_KEY gevonden in .env of omgeving.")
        st.stop()

    client = OpenAI(api_key=api_key)
    data = load_data()

    # Juridische gateway vóór toegang tot app
    if not st.session_state.legal_accepted:
        show_legal_gateway()
        st.stop()

    # SIDEBAR
    with st.sidebar:
        if isinstance(main_logo, str) and os.path.exists(main_logo):
            st.image(main_logo, width=140)

        st.title("Finny Demo")
        st.caption(
            f"Beschikbare jaren: {', '.join(str(y) for y in st.session_state.get('active_years', AVAILABLE_YEARS))}"
        )
        st.markdown("---")

        if st.session_state.user_avatar_path and os.path.exists(
            st.session_state.user_avatar_path
        ):
            st.image(st.session_state.user_avatar_path, width=120, caption="Jij")

        if st.button("Nieuw gesprek", use_container_width=True):
            start_new_conversation()
            st.rerun()

        # Gesprek afronden-knop nu in de sidebar, vlak bij Nieuw gesprek
        if st.button("Gesprek afronden", use_container_width=True):
            st.session_state.conversation_finished = True

        st.markdown("---")
        opts = ["intro", "chat", "history", "share"]
        cur = st.session_state.current_view
        idx = opts.index(cur) if cur in opts else 0
        choice = st.radio(
            "Menu",
            opts,
            index=idx,
            format_func=lambda v: {
                "intro": "Kennismaking",
                "chat": "Chat",
                "history": "Gesprekken",
                "share": "Accountant",
            }[v],
        )
        if choice != st.session_state.current_view:
            st.session_state.current_view = choice
            st.rerun()

        # Privacy-instellingen
        st.markdown("---")
        st.subheader("Privacy & profiel")

        current = st.session_state.profile_consent
        new = st.toggle(
            "Persoonlijk klantprofiel",
            value=current,
            help=(
                "Sta Finny toe uw gegevens te gebruiken voor een persoonlijk klantprofiel "
                "(gepersonaliseerde inzichten op basis van eerdere interacties)."
            ),
        )

        if new != current:
            st.session_state.profile_consent = new
            status = "granted" if new else "revoked"
            log_event(
                CONSENT_LOG,
                f"profile_consent_{status}",
                {"new_value": new},
            )
            st.rerun()

        st.caption(
            "U kunt deze toestemming hier op elk moment intrekken of verlenen. "
            "Wijzigingen worden direct opgeslagen."
        )

    view = st.session_state.current_view

    # INTRO
    if view == "intro":
        st.title("Welkom bij Finny")
        st.write(
            "Ik werk met jouw demo-data (CSV + jaarrekening) en geef korte, feitelijke antwoorden. "
            "Stel een vraag over kosten, winst, omzet of trends."
        )

        prof = st.session_state.client_profile

        with st.form("profiel"):
            kennis = st.slider(
                "Financiële kennis (1 = beginner, 5 = expert)",
                1,
                5,
                int(prof.get("finance_knowledge", 2)),
            )
            focus_val = st.text_input(
                "Belangrijkste focus (bijv. Kostenbesparing, Groei, Rust)",
                prof.get("focus", ""),
            )

            current_detail = prof.get("answer_detail", "Normaal")
            if current_detail not in ["Kort", "Normaal", "Uitgebreid"]:
                current_detail = "Normaal"

            answer_detail = st.select_slider(
                "Hoe uitgebreid wil je dat Finny antwoordt?",
                options=["Kort", "Normaal", "Uitgebreid"],
                value=current_detail,
            )

            uploaded_photo = st.file_uploader(
                "Upload een foto of logo voor dit gesprek (optioneel)",
                type=["png", "jpg", "jpeg"],
            )

            submit_profile = st.form_submit_button("Opslaan & naar chat")

        if submit_profile:
            prof.update(
                {
                    "finance_knowledge": kennis,
                    "focus": focus_val,
                    "answer_detail": answer_detail,
                }
            )

            if uploaded_photo is not None:
                ext = os.path.splitext(uploaded_photo.name)[1].lower() or ".png"
                avatar_path = f"user_avatar{ext}"
                with open(avatar_path, "wb") as f:
                    f.write(uploaded_photo.getbuffer())
                st.session_state.user_avatar_path = avatar_path

            st.session_state.client_profile = prof
            st.session_state.current_view = "chat"
            st.rerun()

    # CHAT
    elif view == "chat":
        st.title("Finny Demo")

        finny_avatar = main_logo if os.path.exists(main_logo) else "🤖"

        if st.session_state.user_avatar_path and os.path.exists(
            st.session_state.user_avatar_path
        ):
            user_avatar = st.session_state.user_avatar_path
        else:
            user_avatar = "👤"

        # Bestaande berichten renderen met alleen escalatie-extra's
        for idx, m in enumerate(st.session_state.messages):
            role = m.get("role", "assistant")
            content = m.get("content", "")
            with st.chat_message(
                role,
                avatar=finny_avatar if role == "assistant" else user_avatar,
            ):
                st.write(content)
                if role == "assistant":
                    render_assistant_extras(content, idx)

        prompt = st.chat_input("Vraag Finny iets over je cijfers...")

        if prompt:
            st.session_state.conversation_finished = False
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user", avatar=user_avatar).write(prompt)

            with st.chat_message("assistant", avatar=finny_avatar):
                with st.spinner("Even nadenken..."):
                    effective_question = prompt
                    pending = st.session_state.get("pending_followup")
                    followup_mode = "NONE"

                    if pending:
                        follow_kind = classify_followup_answer(prompt)
                        if follow_kind == "CONFIRM":
                            topic_override = extract_optional_topic_from_text(prompt)
                            topic = topic_override or pending.get("topic", "winst")

                            years = pending.get("years") or [
                                data.get("latest_year", max(AVAILABLE_YEARS))
                            ]
                            years = [y for y in years if isinstance(y, int)]
                            if years:
                                jaar_label = f" tussen {min(years)} en {max(years)}"
                            else:
                                jaar_label = ""

                            effective_question = (
                                "Welke kostenposten (bijvoorbeeld "
                                f"{topic}) hebben de grootste impact gehad op de "
                                f"ontwikkeling van de winst{jaar_label}?"
                            )
                            st.session_state.pending_followup = None
                        elif follow_kind == "DECLINE":
                            st.session_state.pending_followup = None
                            effective_question = prompt
                        else:
                            st.session_state.pending_followup = None
                            effective_question = prompt

                    if st.session_state.get("pending_followup") is None:
                        synthetic = detect_simple_followup(
                            prompt,
                            st.session_state.get("last_analysis"),
                            data,
                        )
                        if synthetic:
                            if synthetic.startswith("[FOLLOWUP_EXPLAIN]"):
                                followup_mode = "EXPLAIN_ONLY"
                            effective_question = synthetic

                    if followup_mode == "EXPLAIN_ONLY":
                        analysis = {
                            "intent": {"type": "CHAT", "term": ""},
                            "years": [],
                            "source": "GESPREK",
                            "context": "",
                            "total": None,
                        }
                    else:
                        analysis = build_analysis(client, effective_question, data)

                    st.session_state.last_analysis = analysis

                    context = analysis["context"]
                    source = analysis["source"]
                    years = analysis["years"]
                    intent = analysis["intent"]

                    st.caption(
                        f"Bron: {source} | Jaren: {', '.join(str(y) for y in years) or 'n.v.t.'}"
                    )

                    profile = st.session_state.client_profile
                    fin_know = int(profile.get("finance_knowledge", 2))

                    if fin_know <= 2:
                        tone = (
                            "Leg het uit in eenvoudige taal (Jip-en-Janneke). "
                            "Geen moeilijke termen zoals 'EBITDA' zonder uitleg."
                        )
                    elif fin_know >= 4:
                        tone = (
                            "Gebruik professionele financiële termen en wees zakelijk en to-the-point."
                        )
                    else:
                        tone = (
                            "Gebruik normale ondernemerstaal: concreet, zonder overdreven jargon."
                        )

                    detail = profile.get("answer_detail", "Normaal")
                    if detail not in ["Kort", "Normaal", "Uitgebreid"]:
                        detail = "Normaal"
                    if detail == "Kort":
                        base_max_sentences = 1
                    elif detail == "Uitgebreid":
                        base_max_sentences = 4
                    else:
                        base_max_sentences = 2

                    prompt_lower = effective_question.lower()
                    word_count = len(prompt_lower.split())
                    analysis_terms = [
                        "waarom",
                        "hoe ",
                        "analyse",
                        "analyses",
                        "advies",
                        "adviseer",
                        "inschatting",
                        "verklaar",
                        "wat betekent",
                        "gevolgen",
                        "scenario",
                        "prognose",
                        "toelichting",
                    ]
                    is_analytical = any(term in prompt_lower for term in analysis_terms)

                    if is_analytical or word_count > 18:
                        max_sentences = base_max_sentences + 2
                    else:
                        max_sentences = base_max_sentences
                    if max_sentences > 6:
                        max_sentences = 6

                    is_fact_question = (not is_analytical) and word_count <= 18

                    if is_fact_question and detail != "Uitgebreid":
                        rule5 = (
                            "5. Geef geen afsluitende vervolgvraag; beantwoord alleen de vraag.\n"
                        )
                        allow_generic_followup = False
                    else:
                        rule5 = (
                            "5. Je mag afsluiten met maximaal één korte vervolgvraag "
                            "als dat logisch is voor de ondernemer.\n"
                        )
                        allow_generic_followup = True

                    followup_instruction = ""
                    if (
                        followup_mode != "EXPLAIN_ONLY"
                        and intent["type"] == "TREND"
                        and "winst" in effective_question.lower()
                    ):
                        yrs = years or [data.get("latest_year", max(AVAILABLE_YEARS))]
                        yrs = [y for y in yrs if isinstance(y, int)]
                        st.session_state.pending_followup = {
                            "type": "TOP_COSTS_FOR_PROFIT_TREND",
                            "topic": "winst",
                            "years": yrs,
                        }
                        followup_instruction = (
                            "6. Sluit af met de vraag: "
                            "'Wil je dat ik ook laat zien welke kosten hier het meest aan bijdragen?'\n"
                        )
                    else:
                        if not allow_generic_followup:
                            followup_instruction = ""

                    # Alleen eerdere Q/A als er profieltoestemming is
                    if st.session_state.profile_consent:
                        conversation_snippet = build_conversation_snippet(max_turns=2)
                    else:
                        conversation_snippet = ""

                    if followup_mode == "EXPLAIN_ONLY":
                        context_text = (
                            "LET OP: Je mag GEEN nieuwe cijfers uit andere bronnen gebruiken. "
                            "Baseer je alleen op eerder genoemde bedragen in het gesprek.\n\n"
                        )
                    else:
                        context_text = context

                    system_prompt = (
                        "Je bent Finny, een financiële assistent voor ondernemers.\n\n"
                        "GESPREK TOT NU TOE:\n"
                        f"{conversation_snippet or 'Nog geen eerdere vragen.'}\n\n"
                        f"CONTEXT (FEITEN – gebruik dit als waarheid waar van toepassing):\n{context_text}\n\n"
                        f"INTENTIE: type={intent['type']}, term='{intent.get('term', '')}'.\n\n"
                        "REGELS VOOR JE ANTWOORD:\n"
                        f"1. Geef een direct antwoord op de vraag in hooguit {max_sentences} korte zinnen.\n"
                        "2. Noem concrete bedragen als die in de context of in het gesprek staan.\n"
                        "3. Verzin NOOIT cijfers of jaartallen; als iets ontbreekt of onduidelijk is, zeg dat eerlijk.\n"
                        f"4. {tone}\n"
                        f"{rule5}"
                        f"{followup_instruction}"
                    )

                    msgs = [{"role": "system", "content": system_prompt}]
                    msgs.append({"role": "user", "content": effective_question})

                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4.1-mini", messages=msgs
                        )
                        reply = resp.choices[0].message.content
                    except Exception as e:
                        reply = (
                            f"Er ging iets mis bij het ophalen van het antwoord van het model: {e}"
                        )

                    st.write(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )

                    st.session_state.conversation_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "user_question": prompt,
                            "effective_question": effective_question,
                            "analysis": analysis,
                            "answer": reply,
                        }
                    )

                    # Extra's direct tonen voor deze beurt (alleen escalatie)
                    current_idx = len(st.session_state.messages) - 1
                    render_assistant_extras(reply, current_idx)

        # ===== GESPREK AFRONDEN + FEEDBACK =====
        if (
            st.session_state.conversation_finished
            and not st.session_state.session_feedback_given
            and any(m.get("role") == "assistant" for m in st.session_state.messages)
        ):
            st.markdown("---")
            st.markdown("**Heeft dit gesprek je geholpen?**")
            col1, col2 = st.columns(2)
            with col1:
                good = st.button("👍 Ja", key="conv_fb_up")
            with col2:
                bad = st.button("👎 Niet helemaal", key="conv_fb_down")

            if good:
                log_event(
                    FEEDBACK_LOG,
                    "conversation_feedback",
                    {"rating": "up"},
                )
                st.session_state.session_feedback_given = True

            if bad:
                st.session_state.show_conv_feedback_input = True

            if (
                st.session_state.show_conv_feedback_input
                and not st.session_state.session_feedback_given
            ):
                reason = st.text_area(
                    "Waar schoot Finny tekort in dit gesprek?",
                    key="conv_fb_reason",
                )
                if st.button("Verstuur feedback", key="conv_fb_send"):
                    log_event(
                        FEEDBACK_LOG,
                        "conversation_feedback",
                        {
                            "rating": "down",
                            "reason": reason,
                        },
                    )
                    st.session_state.session_feedback_given = True
                    st.session_state.show_conv_feedback_input = False

        # rustige disclaimer onderin, zoals ChatGPT
        st.caption(
            "Finny kan fouten maken. Controleer belangrijke informatie altijd zelf en raadpleeg bij twijfel uw accountant."
        )

    # HISTORY
    elif view == "history":
        st.title("Gesprekken")
        if not st.session_state.conversations:
            st.info("Nog geen gesprekken opgeslagen. Start een nieuw gesprek in de chat.")
        else:
            for conv in reversed(st.session_state.conversations):
                title = conv.get("title", "Gesprek")
                ts = conv.get("timestamp")
                label = (
                    f"{ts.strftime('%d-%m %H:%M')} - {title}"
                    if isinstance(ts, datetime)
                    else title
                )
                with st.expander(label):
                    for m in conv.get("messages", []):
                        rol = m.get("role", "?")
                        cont = m.get("content", "")
                        st.write(f"**{rol}**: {cont}")

    # SHARE
    elif view == "share":
        st.title("Gesprekken voor accountant")
        if not st.session_state.conversations:
            st.info("Er zijn nog geen gesprekken om te delen.")
        else:
            with st.form("share_form"):
                checks = {}
                for i, conv in enumerate(st.session_state.conversations):
                    title = conv.get("title", f"Gesprek {i+1}")
                    checked = conv.get("shared_with_accountant", False)
                    checks[i] = st.checkbox(title, value=checked, key=f"share_{i}")
                submit_share = st.form_submit_button("Opslaan")

            if submit_share:
                for i, v in checks.items():
                    st.session_state.conversations[i]["shared_with_accountant"] = bool(v)
                st.success("Selectie bijgewerkt.")
                st.rerun()
