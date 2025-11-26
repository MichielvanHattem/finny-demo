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
# 0. SETUP & LOGO
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

# AANPASSING: We starten standaard in de 'intro' (Kennismaking/Intake)
if "current_view" not in st.session_state:
    st.session_state.current_view = "intro"
if "client_profile" not in st.session_state:
    st.session_state.client_profile = {}

def check_password():
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
    return st.session_state["password_correct"]

# --- NAVIGATION ---
def start_new_conversation():
    if st.session_state.messages:
        first_q = "Gesprek zonder titel"
        for m in st.session_state.messages:
            if m["role"] == "user":
                first_q = m["content"][:60] + (
                    "..." if len(m["content"]) > 60 else ""
                )
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
        "rgs": None,
        "pdf_text": "",
        "company_context": "",
        "latest_year": 2024,
    }
    def clean_code(val):
        return str(val).split(".")[0].strip()

    # A. Transacties
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", encoding="latin1")
            if "Finny_Year" in df.columns:
                df["Year_Clean"] = df["Finny_Year"].astype(str).str.split(".").str[0]
                valid_years = pd.to_numeric(
                    df["Year_Clean"], errors="coerce"
                ).dropna()
                if not valid_years.empty:
                    data["latest_year"] = int(valid_years.max())
            cols = [
                "Description",
                "AccountName",
                "Finny_GLDescription",
                "Finny_GLCode",
            ]
            existing = [c for c in cols if c in df.columns]
            df["UniversalSearch"] = (
                df[existing].astype(str).agg(" ".join, axis=1).str.lower()
            )
            data["trans"] = df
        except Exception as e:
            st.error(f"Fout CSV: {e}")

    # B. Synoniemen
    if os.path.exists("Finny_Synonyms.csv"):
        try:
            df = pd.read_csv("Finny_Synonyms.csv", sep=";", dtype=str)
            if "Synoniem" in df.columns:
                df["Synoniem_Clean"] = (
                    df["Synoniem"].astype(str).str.lower().str.strip()
                )
            data["syn"] = df
        except Exception:
            pass

    # C. PDF text
    pdf_files = glob.glob("*.pdf")
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            data["pdf_text"] += f"\n--- JAARREKENING {pdf} ---\n{text}"
        except Exception:
            pass

    # D. Profiel TXT
    txt_files = glob.glob("*profiel*.txt") + glob.glob("*profile*.txt")
    for txt in txt_files:
        try:
            with open(txt, "r", encoding="utf-8") as f:
                data["company_context"] += (
                    f"\n--- PROFIEL ({txt}) ---\n{f.read()}\n"
                )
        except Exception:
            try:
                with open(txt, "r", encoding="latin1") as f:
                    data["company_context"] += (
                        f"\n--- PROFIEL ({txt}) ---\n{f.read()}\n"
                    )
            except Exception:
                pass

    return data

# ==========================================
# 3. ROUTER & LOGIC
# ==========================================
def get_intent(client, question, data):
    q_lower = question.lower()
    context_years = st.session_state.get("active_years", AVAILABLE_YEARS)

    # 1. CHECK SYNONIEMEN
    csv_keywords = []
    if (
        data["syn"] is not None
        and not data["syn"].empty
        and "Synoniem_Clean" in data["syn"].columns
    ):
        known_synonyms = data["syn"]["Synoniem_Clean"].dropna().unique()
        for syn in known_synonyms:
            if syn in q_lower and len(syn) > 2:
                csv_keywords.append(syn)

    if csv_keywords:
        if not any(
            x in q_lower for x in ["waarom", "hoe komt", "oorzaak", "verloop"]
        ):
            return {"source": "CSV", "keywords": csv_keywords, "years": context_years}

    # 2. CHECK PDF SIGNALEN
    TREND_TERMS = [
        "stijgen",
        "dalen",
        "toenemen",
        "afnemen",
        "waarom",
        "hoe komt",
        "oorzaak",
        "verloop",
        "trend",
        "verschil",
        "vergelijk",
        "winst",
        "omzet",
        "resultaat",
        "balans",
    ]
    if any(t in q_lower for t in TREND_TERMS):
        return {"source": "PDF", "keywords": [], "years": context_years}

    # 3. LLM FALLBACK
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Je bent een router. Kies 'PDF' voor algemene vragen/trends. Kies 'CSV' voor specifieke transactievragen. Output JSON: {source: 'PDF'|'CSV', years: [], keywords: []}",
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content)
    except Exception:
        return {"source": "PDF", "keywords": [], "years": context_years}

def analyze_csv_costs(data, intent):
    if data["trans"] is None:
        return "Geen transacties beschikbaar."
    df = data["trans"].copy()
    keywords = intent.get("keywords", [])
    years = [str(y) for y in intent.get("years", AVAILABLE_YEARS)]

    if "Year_Clean" in df.columns:
        df = df[df["Year_Clean"].astype(str).isin(years)]
    if "Description" in df.columns:
        df = df[
            ~df["Description"].astype(str).str.contains(
                r"(resultaat|winst|balans)", case=False, na=False
            )
        ]

    found_categories = set()
    df_found = pd.DataFrame()

    if keywords:
        pat = "|".join([re.escape(k.lower()) for k in keywords])
        if "UniversalSearch" in df.columns:
            df_found = df[df["UniversalSearch"].astype(str).str.contains(pat, na=False)]

        if data["syn"] is not None:
            syn_df = data["syn"]
            for k in keywords:
                matches = syn_df[syn_df["Synoniem_Clean"] == k]
                if not matches.empty and "Categorie" in matches.columns:
                    found_categories.update(matches["Categorie"].dropna().unique())

    df_cat = pd.DataFrame()
    if found_categories:
        cat_pat = "|".join([re.escape(c) for c in found_categories])
        if "Finny_GLDescription" in df.columns:
            df_cat = df[
                df["Finny_GLDescription"].astype(str).str.contains(
                    cat_pat, case=False, na=False
                )
            ]

    df_total = pd.concat([df_found, df_cat]).drop_duplicates()

    if df_total.empty:
        return (
            f"Geen transacties gevonden voor: {keywords}. (Gecheckt in synoniemen en transacties)."
        )

    total = df_total["AmountDC_num"].sum() if "AmountDC_num" in df_total.columns else 0

    res = f"### DETAILS UIT TRANSACTIES ({', '.join(years)})\n"
    if found_categories:
        res += f"**Gevonden categorieën:** {', '.join(found_categories)}\n\n"
    res += f"**Totaalbedrag:** € {total:,.2f}\n\n"

    if "Description" in df_total.columns:
        res += "**Top 5 kostenposten:**\n"
        top = (
            df_total.groupby("Description")["AmountDC_num"].sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        res += top.to_markdown(index=False, floatfmt=".2f")
    return res

# ==========================================
# 4. MAIN UI
# ==========================================
if check_password():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.stop()
    client = OpenAI(api_key=api_key)
    data = load_data()

    # SIDEBAR
    with st.sidebar:
        if os.path.exists(main_logo):
            st.image(main_logo, width=150)
        st.title("Finny Demo")
        st.caption(f"Geheugen: {st.session_state.get('active_years', AVAILABLE_YEARS)}")
        st.markdown("---")
        if st.button("Nieuw Gesprek", use_container_width=True):
            start_new_conversation()
            st.rerun()
        st.markdown("---")
        opts = ["intro", "chat", "history", "share"]
        curr_idx = (
            opts.index(st.session_state.current_view)
            if st.session_state.current_view in opts
            else 0
        )
        choice = st.radio(
            "Menu",
            opts,
            index=curr_idx,
            format_func=lambda x: {
                "chat": "Chat",
                "intro": "Kennismaking",
                "history": "Gesprekken",
                "share": "Accountant",
            }[x],
        )
        if choice != st.session_state.current_view:
            st.session_state.current_view = choice
            st.rerun()

    # VIEW ROUTING
    view = st.session_state.current_view

    # VIEW: INTRO
    if view == "intro":
        st.title("Welkom bij Finny")
        st.write(
            "Laten we even kennismaken. Upload je foto en vul je profiel in, dan kan ik je beter helpen."
        )
        curr = st.session_state.client_profile
        up = st.file_uploader("Profielfoto", type=["jpg", "png"])
        if up:
            curr["avatar"] = up
            st.image(up, width=100)
        elif curr.get("avatar"):
            st.image(curr["avatar"], width=100)
        with st.form("prof"):
            fk = st.slider(
                "Financiële kennis (1=Beginner, 5=Expert)",
                1,
                5,
                curr.get("finance_knowledge", 2),
            )
            risk = st.slider(
                "Risicobereidheid", 1, 5, curr.get("risk_preference", 3)
            )
            focus = st.text_input(
                "Focusgebied (bijv. Kostenbesparing)", curr.get("focus_areas", "")
            )
            if st.form_submit_button("Opslaan & Starten"):
                curr.update(
                    {
                        "finance_knowledge": fk,
                        "risk_preference": risk,
                        "focus_areas": focus,
                    }
                )
                st.session_state.client_profile = curr
                st.success("Opgeslagen! Je kunt nu naar de chat.")
                st.session_state.current_view = "chat"
                st.rerun()

    # VIEW: CHAT
    elif view == "chat":
        st.title("Finny Demo")
        finny_av = main_logo if os.path.exists(main_logo) else "🤖"
        user_av = st.session_state.client_profile.get("avatar") or "👤"
        for m in st.session_state.messages:
            st.chat_message(
                m["role"],
                avatar=finny_av if m["role"] == "assistant" else user_av,
            ).write(m["content"])

        if prompt := st.chat_input("Vraag Finny..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user", avatar=user_av).write(prompt)
            with st.chat_message("assistant", avatar=finny_av):
                with st.spinner("..."):
                    intent = get_intent(client, prompt, data)
                    source = intent.get("source", "PDF")
                    context = None
                    caption_txt = None

                    # bepaal of jaartal en categorie ontbreken
                    user_years = intent.get("years", [])
                    missing_year = (not user_years) or (
                        set(user_years) == set(AVAILABLE_YEARS)
                    )
                    missing_cat = False
                    if source == "CSV":
                        missing_cat = not intent.get("keywords")

                    # bepaal een geschikt voorbeeldjaar
                    latest_year = data.get("latest_year", None)
                    if latest_year is not None and (latest_year in AVAILABLE_YEARS):
                        suggest_year = latest_year
                    else:
                        suggest_year = max(AVAILABLE_YEARS)

                    # als informatie ontbreekt: stel een vervolgvraag
                    if missing_year or (source == "CSV" and missing_cat):
                        reply_lines = ["Om je goed te helpen heb ik nog wat informatie nodig:"]
                        if source == "CSV" and missing_year and missing_cat:
                            reply_lines.append(
                                "Voor welke periode en over welke soort kosten gaat je vraag precies?"
                            )
                        elif missing_year:
                            reply_lines.append(
                                "Voor welke periode bedoel je dit precies?"
                            )
                        else:  # alleen categorie ontbreekt
                            reply_lines.append(
                                "Over welke soort kosten of categorie gaat je vraag precies?"
                            )
                        hints = []
                        if missing_year and suggest_year:
                            hints.append(
                                f"– Bijvoorbeeld: 'Wat was mijn winst in {suggest_year}?'"
                            )
                            if source == "CSV":
                                hints.append(
                                    f"– Of: 'Hoeveel autokosten had ik in {suggest_year}?'"
                                )
                        if source == "CSV" and missing_cat:
                            hints.append(
                                "– Bijvoorbeeld: autokosten, telefoonkosten, huisvesting of personeel."
                            )
                        if hints:
                            reply_lines.append("")
                            reply_lines.extend(hints)
                        reply = "\n".join(reply_lines)
                        st.write(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )
                    else:
                        # informatie compleet: bouw context
                        if source == "CSV":
                            context = analyze_csv_costs(data, intent)
                            caption_txt = f"Bron: Transacties (Details) | Zoektermen: {intent.get('keywords')}"
                        else:
                            context = data["pdf_text"]
                            caption_txt = (
                                f"Bron: Jaarrekening (Trends) | Jaren: {intent.get('years')}"
                            )

                    # Alleen verder gaan als er context is (dus geen vraag nodig)
                    if context:
                        st.caption(caption_txt)
                        profile = st.session_state.client_profile
                        fin_know = profile.get("finance_knowledge", 3)
                        tone_instruction = (
                            "Gebruik Jip-en-Janneke taal. Vermijd jargon zoals 'immateriële activa', zeg gewoon 'bezittingen'."
                        )
                        if fin_know >= 4:
                            tone_instruction = (
                                "Gebruik professionele financiële termen. Wees zakelijk."
                            )
                        company_info = (
                            data["company_context"] or "Geen bedrijfsprofiel."
                        )
                        system_prompt = f"""
                        Je bent Finny. Je helpt ondernemers.

                        BEDRIJFSPROFIEL:
                        {company_info}

                        DATA (Gebruik dit!):
                        {context}

                        REGELS:
                        1. TOON: {tone_instruction}
                        2. FORMAT: Max 3-4 zinnen per antwoord. Gebruik bullets. KORT & BONDIG.
                        3. DATA: Verzin nooit cijfers.
                        4. EINDE: Eindig ALTIJD met een relevante vervolgvraag.
                        """
                        msgs = [{"role": "system", "content": system_prompt}]
                        for m in st.session_state.messages[-3:]:
                            msgs.append(
                                {
                                    "role": "user" if m["role"] == "user" else "assistant",
                                    "content": m["content"],
                                }
                            )
                        res = client.chat.completions.create(
                            model="gpt-4o-mini", messages=msgs
                        )
                        reply = res.choices[0].message.content
                        st.write(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )

    elif view == "history":
        st.title("Gesprekken")
        for c in reversed(st.session_state.conversations):
            with st.expander(
                f"{c['timestamp'].strftime('%d-%m %H:%M')} - {c['title']}"
            ):
                for m in c["messages"]:
                    st.write(f"**{m['role']}**: {m['content']}")

    elif view == "share":
        st.title("Accountant")
        if not st.session_state.conversations:
            st.info("Geen gesprekken.")
        else:
            with st.form("sh"):
                sel = {}
                for i, c in enumerate(st.session_state.conversations):
                    sel[i] = st.checkbox(
                        f"{c['timestamp'].strftime('%H:%M')} - {c['title']}",
                        value=c.get("shared", False),
                        key=i,
                    )
                if st.form_submit_button("Opslaan"):
                    for i, v in sel.items():
                        st.session_state.conversations[i]["shared"] = v
                    st.success("Gemarkeerd.")
                    st.rerun()
