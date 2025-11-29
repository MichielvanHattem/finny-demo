import streamlit as st
import pandas as pd
import os, re, glob, json
from datetime import datetime
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 0. SETUP & CONFIG
# ==========================================
logo_files = glob.glob("finny_logo.png") + glob.glob("*.png") + glob.glob("*.jpg")
main_logo = "finny_logo.png" if os.path.exists("finny_logo.png") else (logo_files[0] if logo_files else "💰")

st.set_page_config(page_title="Finny Demo", page_icon=main_logo, layout="wide")
load_dotenv()

AVAILABLE_YEARS = [2022, 2023, 2024]

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
                    # geen toekomstjaren in demo
                    data["latest_year"] = min(latest, max(AVAILABLE_YEARS))
                    st.session_state["active_years"] = sorted(
                        y for y in AVAILABLE_YEARS if y <= data["latest_year"]
                    )
            # noodzakelijke kolommen
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
        except Exception as e:
            st.error(f"Fout bij laden Finny_Synonyms.csv: {e}")

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
# 3. SANDWICH-ARCHITECTUUR
#    Code → AI (intent) → Code (data) → AI (antwoord)
# ==========================================
def classify_intent(client: OpenAI, question: str) -> dict:
    """
    AI-intent: bepaal type vraag en hoofdterm.
    type ∈ {TOTAL_COST, SPECIFIC_COST, TREND, CHAT}
    LET OP: géén jaartal-detectie hier. :contentReference[oaicite:1]{index=1}
    """
    system = (
        "Je bent een router voor de Finny demo.\n"
        "Analyseer de vraag van een ondernemer en geef ALLEEN JSON terug met:\n"
        "{\n"
        '  "type": "TOTAL_COST" | "SPECIFIC_COST" | "TREND" | "CHAT",\n'
        '  "term": "<hoofdonderwerp of kostenpost, leeg bij TOTAL/CHAT>"\n'
        "}\n\n"
        "- TOTAL_COST: totaal van kosten/uitgaven/resultaat in een periode.\n"
        "- SPECIFIC_COST: specifieke post zoals telefoon, auto, huur, personeel.\n"
        "- TREND: verloop/vergelijking over jaren, stijging/daling.\n"
        "- CHAT: uitleg, begroetingen, algemene vragen.\n"
        "Gebruik Nederlandse termen zoals de gebruiker.\n"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        t = data.get("type", "CHAT")
        term = (data.get("term") or "").strip()
        if t not in {"TOTAL_COST", "SPECIFIC_COST", "TREND", "CHAT"}:
            t = "CHAT"
        return {"type": t, "term": term}
    except Exception:
        return {"type": "CHAT", "term": ""}


def extract_years(question: str, latest_year: int, intent_type: str) -> list[int]:
    """
    Jaartallen pas ná intent detecteren; anders dan in de mislukte versies. :contentReference[oaicite:2]{index=2}
    """
    years = sorted(
        {int(y) for y in re.findall(r"\b(20[0-9]{2})\b", question) if int(y) in AVAILABLE_YEARS}
    )
    if years:
        return years
    # geen jaartal: maak aannames op basis van type
    if intent_type in {"TOTAL_COST", "SPECIFIC_COST"}:
        return [latest_year]
    if intent_type == "TREND":
        yrs = [y for y in AVAILABLE_YEARS if y <= latest_year]
        return yrs[-3:] if len(yrs) >= 3 else yrs
    return []


def build_csv_query(data: dict, intent: dict, years: list[int]) -> tuple[str | None, float | None]:
    df = data.get("trans")
    syn = data.get("syn")
    if df is None or df.empty:
        return None, None

    str_years = [str(y) for y in years] if years else None
    if str_years and "Year_Clean" in df.columns:
        df = df[df["Year_Clean"].astype(str).isin(str_years)]

    # filter technische afsluitboekingen (resultaat naar balans) :contentReference[oaicite:3]{index=3}
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

    intent_type = intent["type"]
    term = (intent["term"] or "").lower().strip()
    filtered = df

    if intent_type == "SPECIFIC_COST" and term:
        patterns: list[str] = [re.escape(term)]
        # synoniemen uit externe CSV (geen hardcoded mapping) :contentReference[oaicite:4]{index=4}
        if syn is not None and "Synoniem_Clean" in syn.columns:
            mask = syn["Synoniem_Clean"].astype(str).str.contains(term, case=False, na=False)
            syn_terms = syn.loc[mask, "Synoniem_Clean"].dropna().unique().tolist()
            patterns.extend(re.escape(t) for t in syn_terms)
            for col in ["Categorie", "Finny_GLDescription", "Finny_GLCode"]:
                if col in syn.columns:
                    patterns.extend(
                        re.escape(str(v).lower())
                        for v in syn.loc[mask, col].dropna().unique().tolist()
                    )
        pat = "|".join(sorted(set(patterns)))
        if "UniversalSearch" in df.columns and pat:
            filtered = df[df["UniversalSearch"].astype(str).str.contains(pat, na=False)]
    elif intent_type == "TOTAL_COST":
        filtered = df
    elif intent_type == "TREND":
        # zelfde selectie als TOTAL, later per jaar groeperen
        filtered = df

    if filtered.empty:
        return None, None

    total = float(filtered["AmountDC_num"].sum())
    if abs(total) < 1e-6:
        # QC: saldo 0 is verdacht → laat AI liever PDF gebruiken
        return None, None

    yrs_label = ", ".join(str(y) for y in years) if years else "alle jaren"
    lines: list[str] = []

    if intent_type == "TOTAL_COST":
        lines.append(f"### TOTALE KOSTEN ({yrs_label})")
        lines.append(f"Totaal: € {total:,.2f}")
    elif intent_type == "SPECIFIC_COST":
        label = term or "geselecteerde kosten"
        lines.append(f"### SPECIFIEKE KOSTEN – {label} ({yrs_label})")
        lines.append(f"Totaal: € {total:,.2f}")
    elif intent_type == "TREND":
        lines.append(f"### VERLOOP ({yrs_label}) – Samenvatting uit transacties")
        if "Year_Clean" in filtered.columns:
            per_year = (
                filtered.groupby("Year_Clean")["AmountDC_num"]
                .sum()
                .reset_index()
                .sort_values("Year_Clean")
            )
            lines.append("")
            lines.append(per_year.to_markdown(index=False, floatfmt=".2f"))
        else:
            lines.append(f"Totaal alle jaren samen: € {total:,.2f}")
    else:
        lines.append(f"### CIJFERS ({yrs_label})")
        lines.append(f"Totaal: € {total:,.2f}")

    if "Description" in filtered.columns:
        top = (
            filtered.groupby("Description")["AmountDC_num"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        if not top.empty:
            lines.append("")
            lines.append("Top 5 posten:")
            lines.append(top.to_markdown(index=False, floatfmt=".2f"))

    context = "\n".join(lines)
    return context, total


def build_analysis(client: OpenAI, question: str, data: dict) -> dict:
    """
    Volledige sandwich:
    1) AI-intent (type/term)
    2) Jaar-detectie (code)
    3) CSV-query (code) + QC
    4) Fallback naar PDF indien nodig
    """
    intent = classify_intent(client, question)  # eerst intent, dan jaar :contentReference[oaicite:5]{index=5}
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


# ==========================================
# 4. MAIN UI
# ==========================================
if check_password():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("Geen OPENAI_API_KEY gevonden in .env of Streamlit secrets.")
        st.stop()
    client = OpenAI(api_key=api_key)
    data = load_data()

    # SIDEBAR
    with st.sidebar:
        if os.path.exists(main_logo):
            st.image(main_logo, width=140)
        st.title("Finny Demo")
        st.caption(
            f"Beschikbare jaren: {', '.join(str(y) for y in st.session_state.get('active_years', AVAILABLE_YEARS))}"
        )
        st.markdown("---")
        if st.button("Nieuw gesprek", use_container_width=True):
            start_new_conversation()
            st.rerun()
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
            focus = st.text_input(
                "Belangrijkste focus (bijv. Kostenbesparing, Groei, Rust)",
                prof.get("focus", ""),
            )
            if st.form_submit_button("Opslaan & naar chat"):
                prof.update({"finance_knowledge": kennis, "focus": focus})
                st.session_state.client_profile = prof
                st.session_state.current_view = "chat"
                st.rerun()

    # CHAT
    elif view == "chat":
        st.title("Finny Demo")
        finny_avatar = main_logo if os.path.exists(main_logo) else "🤖"
        user_avatar = st.session_state.client_profile.get("avatar") or "👤"

        for m in st.session_state.messages:
            st.chat_message(
                m["role"],
                avatar=finny_avatar if m["role"] == "assistant" else user_avatar,
            ).write(m["content"])

        if prompt := st.chat_input("Vraag Finny iets over je cijfers..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user", avatar=user_avatar).write(prompt)

            with st.chat_message("assistant", avatar=finny_avatar):
                with st.spinner("Even rekenen..."):
                    analysis = build_analysis(client, prompt, data)
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
                        tone = "Gebruik normale ondernemerstaal: concreet, zonder overdreven jargon."

                    system_prompt = (
                        "Je bent Finny, een financiële assistent voor ondernemers.\n\n"
                        f"CONTEXT (FEITEN – gebruik dit als waarheid):\n{context}\n\n"
                        f"INTENTIE: type={intent['type']}, term='{intent['term']}'.\n\n"
                        "REGELS VOOR JE ANTWOORD:\n"
                        "1. Geef een direct antwoord op de vraag, in maximaal 4 zinnen.\n"
                        "2. Noem concrete bedragen als die in de context staan.\n"
                        "3. Verzin NOOIT cijfers of jaartallen; als iets ontbreekt, zeg dat eerlijk.\n"
                        "4. "
                        + tone
                        + "\n"
                        "5. Sluit af met één korte vervolgvraag die logisch is voor de ondernemer.\n"
                    )

                    msgs = [{"role": "system", "content": system_prompt}]
                    msgs.append({"role": "user", "content": prompt})

                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini", messages=msgs
                        )
                        reply = resp.choices[0].message.content
                    except Exception as e:
                        reply = f"Er ging iets mis bij het ophalen van het antwoord van het model: {e}"

                    st.write(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
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
                if st.form_submit_button("Opslaan"):
                    for i, v in checks.items():
                        st.session_state.conversations[i][
                            "shared_with_accountant"
                        ] = bool(v)
                    st.success("Selectie bijgewerkt.")
                    st.rerun()

# --- EINDE finny_app.py ---
