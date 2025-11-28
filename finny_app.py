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
# 3. ROUTER & LOGIC (SLIMMER GEMAAKT)
# ==========================================

def get_intent(client, question, data):
    """
    Bepaalt:
    - source: 'PDF' of 'CSV'
    - scope: 'specific' (zoekwoorden), 'total' (alles optellen), 'general' (uitleg)
    - years: lijst met jaren
    - keywords: zoekwoorden voor CSV (alleen als scope specific is)
    - needs_year: of een concreet jaar echt nodig is
    - year_mode: 'single' | 'multi' | 'none'
    """
    q_lower = question.lower()
    context_years = st.session_state.get("active_years", AVAILABLE_YEARS)

    # 1. CHECK SYNONIEMEN (Directe CSV match = Altijd Specific)
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
        # Als er concrete synoniemen zijn, is het een specifieke vraag
        if not any(
            x in q_lower for x in ["waarom", "hoe komt", "oorzaak", "verloop", "vergelijk"]
        ):
            return {
                "source": "CSV",
                "scope": "specific",
                "keywords": csv_keywords,
                "years": context_years,
                "needs_year": True,
                "year_mode": "single",
            }

    # 2. CHECK PDF SIGNALEN (Trends/Verloop = General/Multi)
    TREND_TERMS = [
        "stijgen", "dalen", "toenemen", "afnemen", "waarom", "hoe komt",
        "oorzaak", "verloop", "trend", "verschil", "vergelijk", "ontwikkeling"
    ]
    if any(t in q_lower for t in TREND_TERMS):
        return {
            "source": "PDF",
            "scope": "general",
            "keywords": [],
            "years": context_years,
            "needs_year": False, 
            "year_mode": "multi",
        }

    # 3. LLM FALLBACK – INTELLIGENTE ROUTER
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
Je bent een router voor Finny. Analyseer de vraag van de ondernemer.

Bepaal de volgende velden (JSON output):

1. "source": 
   - 'PDF' voor vragen over trends, algemene balansposten, winst/verlies uitleg.
   - 'CSV' voor vragen over kosten, uitgaven, bedragen of transacties.

2. "scope": (BELANGRIJK)
   - 'specific': De gebruiker zoekt een SPECIFIEKE post (bijv. "autokosten", "huur", "telefoon").
   - 'total': De gebruiker vraagt naar het TOTAAL of ALLES (bijv. "wat zijn mijn kosten", "totale uitgaven", "hoeveel heb ik betaald").
   - 'general': Voor algemene vragen (bijv. "hoe werkt dit", "uitleg", "hallo").

3. "keywords":
   - Alleen invullen als scope = 'specific'. Geef de zelfstandige naamwoorden (bijv. ["auto", "brandstof"]).
   - Laat leeg [] als scope = 'total' of 'general'.

4. "needs_year": true/false (Is een specifiek jaartal nodig voor een correct antwoord?)
5. "year_mode": 'single' (één jaar), 'multi' (verloop/trend), of 'none'.
6. "years": Lijst met jaren als ze expliciet in de tekst staan (bijv. [2023]). Anders leeglaten.

Geef ALLEEN JSON terug.
""",
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        intent = json.loads(res.choices[0].message.content)

        # Fallback waarden
        if not intent.get("years"):
            intent["years"] = context_years
        if "needs_year" not in intent:
            intent["needs_year"] = False
        if "scope" not in intent:
            intent["scope"] = "general"
        if "year_mode" not in intent:
            intent["year_mode"] = "none"

        return intent
    except Exception:
        # Conservatief fallback
        return {
            "source": "PDF",
            "scope": "general",
            "keywords": [],
            "years": context_years,
            "needs_year": False,
            "year_mode": "none",
        }

def analyze_csv_costs(data, intent):
    if data["trans"] is None:
        return "Geen transacties beschikbaar."
    
    df = data["trans"].copy()
    keywords = intent.get("keywords", [])
    scope = intent.get("scope", "general") # specific vs total
    years = [str(y) for y in intent.get("years", AVAILABLE_YEARS)]

    # 1. Filter op jaar
    if "Year_Clean" in df.columns:
        df = df[df["Year_Clean"].astype(str).isin(years)]
    
    # 2. Filter resultaat/balans weg (technisch)
    if "Description" in df.columns:
        df = df[
            ~df["Description"].astype(str).str.contains(
                r"(resultaat|winst|balans)", case=False, na=False
            )
        ]

    # 3. Filter op Keywords (ALLEEN als scope specific is)
    df_found = pd.DataFrame()
    found_categories = set()
    
    if scope == "specific" and keywords:
        # Zoek in omschrijvingen
        pat = "|".join([re.escape(k.lower()) for k in keywords])
        if "UniversalSearch" in df.columns:
            df_found = df[df["UniversalSearch"].astype(str).str.contains(pat, na=False)]

        # Zoek in synoniemen
        if data["syn"] is not None:
            syn_df = data["syn"]
            for k in keywords:
                matches = syn_df[syn_df["Synoniem_Clean"] == k]
                if not matches.empty and "Categorie" in matches.columns:
                    found_categories.update(matches["Categorie"].dropna().unique())
        
        # Zoek in categorieën (GLDescription)
        df_cat = pd.DataFrame()
        if found_categories:
            cat_pat = "|".join([re.escape(c) for c in found_categories])
            if "Finny_GLDescription" in df.columns:
                df_cat = df[
                    df["Finny_GLDescription"].astype(str).str.contains(
                        cat_pat, case=False, na=False
                    )
                ]
        
        # Combineer resultaten
        df_total = pd.concat([df_found, df_cat]).drop_duplicates()
        
    else:
        # Scope is 'total' (of general in csv context) -> Pak alles wat overblijft
        df_total = df.copy()

    # Resultaat opbouwen
    if df_total.empty:
        return f"Geen transacties gevonden."

    total = df_total["AmountDC_num"].sum() if "AmountDC_num" in df_total.columns else 0

    res = f"### DETAILS UIT TRANSACTIES ({', '.join(years)})\n"
    if found_categories:
        res += f"**Gevonden categorieën:** {', '.join(found_categories)}\n\n"
    elif scope == "total":
        res += "**Overzicht totale kosten/transacties:**\n\n"
        
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
                    # 1. Intent & Scope bepalen
                    intent = get_intent(client, prompt, data)
                    source = intent.get("source", "PDF")
                    scope = intent.get("scope", "general")
                    
                    context = None
                    caption_txt = None

                    # 2. Expliciete jaren uit Regex halen (altijd leidend)
                    year_matches = re.findall(r"\b(20[0-9]{2})\b", prompt)
                    explicit_years = [
                        int(y) for y in year_matches if int(y) in AVAILABLE_YEARS
                    ]
                    if explicit_years:
                        intent["years"] = explicit_years

                    # 3. Checken op ontbrekende info
                    user_years = intent.get("years", [])
                    needs_year = bool(intent.get("needs_year", False))
                    year_mode = intent.get("year_mode", "none")

                    # Check 1: Ontbrekend jaar? 
                    # Alleen als: Jaar is nodig + Mode is Single + Geen jaar gevonden
                    missing_year = needs_year and (year_mode == "single") and (
                        not explicit_years
                    )

                    # Check 2: Ontbrekende categorie?
                    # Alleen als: Bron is CSV + Scope is SPECIFIC + Geen keywords
                    # (Dus bij Scope='total' is missing_cat FALSE -> Grote verbetering!)
                    missing_cat = False
                    if source == "CSV" and scope == "specific" and not intent.get("keywords"):
                        missing_cat = True

                    # Suggestie voor jaar (voor in de hulpstekst)
                    latest_year = data.get("latest_year", 2024)
                    suggest_year = latest_year if latest_year in AVAILABLE_YEARS else max(AVAILABLE_YEARS)

                    # 4. Actie: Vragen om info OF antwoord genereren
                    if missing_year or missing_cat:
                        reply_lines = ["Om je goed te helpen heb ik nog wat informatie nodig:"]
                        
                        if missing_year and missing_cat:
                            reply_lines.append("Voor welk jaar en over welke kostenpost gaat dit precies?")
                        elif missing_year:
                            reply_lines.append(f"Over welk jaar gaat je vraag? (Ik heb cijfers t/m {latest_year})")
                        elif missing_cat:
                            reply_lines.append("Over welke specifieke kosten of categorie gaat je vraag?")
                        
                        # Hints toevoegen
                        hints = []
                        if missing_year:
                            hints.append(f"– Bijvoorbeeld: 'in {suggest_year}'")
                        if missing_cat:
                            hints.append("– Bijvoorbeeld: 'autokosten', 'huisvesting' of 'personeel'")
                        
                        if hints:
                            reply_lines.append("")
                            reply_lines.extend(hints)

                        reply = "\n".join(reply_lines)
                        st.write(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )
                    else:
                        # Info is compleet (of scope is 'total'/'general'), we gaan context bouwen
                        if source == "CSV":
                            context = analyze_csv_costs(data, intent)
                            if scope == "total":
                                caption_txt = f"Bron: CSV (Totaaloverzicht) | Jaren: {intent.get('years')}"
                            else:
                                caption_txt = f"Bron: CSV (Detail) | Zoektermen: {intent.get('keywords')}"
                        else:
                            context = data["pdf_text"]
                            caption_txt = (
                                f"Bron: Jaarrekening (Trends) | Jaren: {intent.get('years')}"
                            )

                    # 5. Genereren antwoord met context
                    if context:
                        st.caption(caption_txt)
                        profile = st.session_state.client_profile
                        fin_know = profile.get("finance_knowledge", 3)
                        
                        tone_instruction = "Gebruik Jip-en-Janneke taal. Vermijd jargon."
                        if fin_know >= 4:
                            tone_instruction = "Gebruik professionele financiële termen. Wees zakelijk."
                            
                        company_info = data["company_context"] or "Geen bedrijfsprofiel."
                        
                        system_prompt = f"""
                        Je bent Finny, een slimme financiële assistent.

                        BEDRIJFSPROFIEL:
                        {company_info}

                        GEVONDEN DATA:
                        {context}

                        JOUW OPDRACHT:
                        1. Beantwoord de vraag op basis van de DATA.
                        2. TOON: {tone_instruction}
                        3. FORMAT: Max 3-4 zinnen. Gebruik bullets voor opsommingen.
                        4. Geef concrete bedragen als die in de data staan. Verzin NOOIT cijfers.
                        5. Eindig met een behulpzame vervolgvraag.
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
