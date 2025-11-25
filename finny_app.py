import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader

# ==========================================
# 1. CONFIGURATIE & CONSTANTEN
# ==========================================

st.set_page_config(page_title="Finny", layout="wide")
load_dotenv()

# Beschikbare jaren in deze demo
AVAILABLE_YEARS = [2022, 2023, 2024]

# Wachtwoord voor demo-toegang
DEMO_PASSWORD = "demo2025"

# ==========================================
# 2. SESSION STATE INITIALISATIE
# ==========================================

if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if "active_years" not in st.session_state:
    # Start met alle beschikbare jaren als context
    st.session_state.active_years = AVAILABLE_YEARS.copy()

if "messages" not in st.session_state:
    # Chatgeschiedenis van de huidige sessie
    st.session_state.messages: List[Dict[str, str]] = []

if "conversations" not in st.session_state:
    # Logging van eerdere gesprekken
    st.session_state.conversations: List[Dict[str, Any]] = []

if "current_view" not in st.session_state:
    # chat, history, share, intro
    st.session_state.current_view = "chat"

if "client_profile" not in st.session_state:
    st.session_state.client_profile = None


# ==========================================
# 3. AUTHENTICATIE
# ==========================================

def check_password() -> bool:
    """Eenvoudige wachtwoordcheck voor de demo."""
    if st.session_state.password_correct:
        return True

    def _submit():
        if st.session_state.get("password_input", "") == DEMO_PASSWORD:
            st.session_state.password_correct = True
        else:
            st.session_state.password_correct = False

    st.text_input(
        "Wachtwoord",
        type="password",
        key="password_input",
        on_change=_submit,
    )
    if st.session_state.password_correct:
        st.success("Toegang verleend.")
        return True
    else:
        st.warning("Voer het demo-wachtwoord in.")
        return False


# ==========================================
# 4. DATA LADEN
# ==========================================

@st.cache_data(ttl=3600)
def load_data() -> Dict[str, Any]:
    """
    Laadt CSV-bestanden en PDF-teksten.
    Verwacht Finny-transacties, synoniemen en RGS-bestand in de huidige directory.
    """
    data: Dict[str, Any] = {
        "trans": None,
        "syn": None,
        "rgs": None,
        "pdf_text": "",
        "latest_year": max(AVAILABLE_YEARS),
    }

    def clean_code(val: Any) -> str:
        return str(val).split(".")[0].strip()

    # A. Transacties
    trans_path = "Finny_Transactions.csv"
    if os.path.exists(trans_path):
        try:
            df_t = pd.read_csv(trans_path, sep=";", dtype=str, encoding="latin1")
            # Normaliseer kolommen
            df_t.columns = [c.strip() for c in df_t.columns]

            # Finny_Year schoonmaken
            if "Finny_Year" in df_t.columns:
                df_t["Year_Clean"] = (
                    df_t["Finny_Year"].astype(str).str.split(".").str[0]
                )
                valid_years = pd.to_numeric(
                    df_t["Year_Clean"], errors="coerce"
                ).dropna()
                if not valid_years.empty:
                    data["latest_year"] = int(valid_years.max())

            # AmountDC_num naar numeriek
            if "AmountDC_num" in df_t.columns:
                df_t["AmountDC_num"] = pd.to_numeric(
                    df_t["AmountDC_num"].astype(str)
                    .str.replace(",", ".", regex=False),
                    errors="coerce",
                ).fillna(0.0)

            # UniversalSearch kolom voor vrije tekst
            us_cols = [
                c
                for c in [
                    "Description",
                    "AccountName",
                    "Finny_GLDescription",
                    "Finny_GLCode",
                ]
                if c in df_t.columns
            ]
            if us_cols:
                df_t["UniversalSearch"] = (
                    df_t[us_cols].astype(str).agg(" ".join, axis=1).str.lower()
                )
            else:
                df_t["UniversalSearch"] = ""

            # Finny_GLCode schoonmaken
            if "Finny_GLCode" in df_t.columns:
                df_t["Finny_GLCode"] = df_t["Finny_GLCode"].apply(clean_code)

            data["trans"] = df_t
        except Exception as e:  # pragma: no cover
            st.error(f"Fout bij laden transacties: {e}")

    # B. Synoniemen
    syn_path = "Finny_Synonyms.csv"
    if os.path.exists(syn_path):
        try:
            df_s = pd.read_csv(syn_path, sep=";", dtype=str, encoding="latin1")
            df_s.columns = [c.strip() for c in df_s.columns]
            # Verwachte kolommen: Synoniem, Categorie, Finny_GLCode
            if "Finny_GLCode" in df_s.columns:
                df_s["Finny_GLCode"] = df_s["Finny_GLCode"].apply(clean_code)
            data["syn"] = df_s
        except Exception as e:  # pragma: no cover
            st.error(f"Fout bij laden synoniemen: {e}")

    # C. RGS
    rgs_path = "Finny_RGS.csv"
    if os.path.exists(rgs_path):
        try:
            df_r = pd.read_csv(rgs_path, sep=";", dtype=str, encoding="latin1")
            df_r.columns = [c.strip() for c in df_r.columns]
            cols = [c for c in df_r.columns if c.startswith("RGS")]
            if cols:
                df_r["SearchBlob"] = (
                    df_r[cols].astype(str).agg(" ".join, axis=1).str.lower()
                )
            data["rgs"] = df_r
        except Exception as e:  # pragma: no cover
            st.error(f"Fout bij laden RGS: {e}")

    # D. PDF-jaarrekeningen
    pdf_texts: List[str] = []
    for fname in os.listdir("."):
        if fname.lower().endswith(".pdf"):
            try:
                reader = PdfReader(fname)
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    pdf_texts.append(txt)
            except Exception:
                continue
    data["pdf_text"] = "\n\n".join(pdf_texts)

    return data


# ==========================================
# 5. ROUTER & CSV-ANALYSE
# ==========================================

def get_intent(client: OpenAI, question: str) -> Dict[str, Any]:
    """
    Vraagt aan het model of een vraag beter via PDF of CSV kan worden beantwoord
    en welke jaren / zoekwoorden daarbij horen.
    """
    context_years = st.session_state.get("active_years", AVAILABLE_YEARS)

    system_prompt = f"""
Je bent de router van Finny.
Je beslist alleen over METADATA, niet over het uiteindelijke antwoord.

CONTEXT:
- Beschikbare jaren in deze demo: {AVAILABLE_YEARS}
- Huidige focusjaren in de chat: {context_years}

TAAK:
Analyseer de gebruikersvraag en geef een JSON terug met:
- "source": "PDF" of "CSV"
  * PDF: winst/omzet/totale kosten, balansachtige vragen.
  * CSV: specifieke kosten (telefoon, communicatie, auto, kantoorkosten), leveranciers, transacties, trends.
- "years": lijst van jaren (bijv. [2024] of [2022, 2023, 2024]).
  * Als de vraag geen jaar noemt, gebruik de focusjaren {context_years}.
- "keywords": lijst van kernwoorden voor CSV-zoekwerk (bijv. ["telefoon", "vodafone"]).

BELANGRIJK:
- Gebruik alleen jaren uit {AVAILABLE_YEARS}. Als je een ander jaar denkt te zien, negeer dat.
De output MOET geldige JSON zijn.
    """.strip()

    user_msg = f"Vraag: {question}"
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        intent_raw = res.choices[0].message.content
        intent = json.loads(intent_raw)
    except Exception:
        # Fallback: alles via PDF
        intent = {"source": "PDF", "years": context_years, "keywords": []}

    # Normaliseer jaren en houd ze binnen AVAILABLE_YEARS
    years_raw = intent.get("years") or context_years
    norm_years: List[int] = []
    for y in years_raw:
        try:
            yi = int(str(y))
            if yi in AVAILABLE_YEARS:
                norm_years.append(yi)
        except ValueError:
            continue
    if not norm_years:
        norm_years = context_years
    intent["years"] = norm_years
    st.session_state.active_years = norm_years

    # Keywords altijd lijst van strings
    kws = intent.get("keywords") or []
    intent["keywords"] = [str(k).lower() for k in kws]

    # Extra harde regel: winst/omzet/totale kosten ALTIJD via PDF
    q_lower = question.lower()
    if any(w in q_lower for w in ["winst", "resultaat", "omzet", "totale kosten"]):
        intent["source"] = "PDF"

    return intent


def analyze_csv_costs(data: Dict[str, Any], intent: Dict[str, Any]) -> str:
    """
    Maakt een CSV-gebaseerde analyse voor kosten/omzetvragen.
    Geeft een markdown-string met tabellen terug.
    """
    df = data.get("trans")
    syn = data.get("syn")
    if df is None or df.empty:
        return "Geen transactiedata beschikbaar."

    years = [str(y) for y in intent.get("years", AVAILABLE_YEARS)]
    keywords = intent.get("keywords", [])

    # Filter op jaar
    if "Year_Clean" in df.columns:
        df_year = df[df["Year_Clean"].isin(years)].copy()
    elif "Finny_Year" in df.columns:
        df_year = df[df["Finny_Year"].astype(str).isin(years)].copy()
    else:
        df_year = df.copy()

    if df_year.empty:
        return f"Geen transacties gevonden voor jaren {years}."

    # Verwijder technische boekingen
    if "Description" in df_year.columns:
        mask_tech = df_year["Description"].astype(str).str.contains(
            r"(resultaat|balans|afsluiting)",
            case=False,
            na=False,
        )
        df_year = df_year[~mask_tech]

    # Zoek categorieën op basis van synoniemen
    found_categories: List[str] = []
    if syn is not None and not syn.empty and keywords:
        if "Synoniem" in syn.columns and "Categorie" in syn.columns:
            for k in keywords:
                m = syn[syn["Synoniem"].astype(str).str.lower().str.contains(k)]
                if not m.empty:
                    found_categories.extend(
                        [c for c in m["Categorie"].dropna().unique().tolist()]
                    )

    found_categories = sorted(set(found_categories))

    res_lines: List[str] = []
    res_lines.append(f"### Analyse transacties voor jaren {', '.join(years)}")

    # Specifieke zoekopdracht (keywords) in UniversalSearch
    if keywords and "UniversalSearch" in df_year.columns:
        pattern = "|".join(re.escape(k) for k in keywords)
        df_spec = df_year[df_year["UniversalSearch"].str.contains(pattern, na=False)]
    else:
        df_spec = pd.DataFrame()

    if not df_spec.empty and "AmountDC_num" in df_spec.columns:
        pivot_spec = (
            df_spec.groupby("Year_Clean")["AmountDC_num"].sum().reset_index()
        )
        pivot_spec.columns = ["Jaar", f'Specifiek: "{", ".join(keywords)}"']
        res_lines.append(pivot_spec.to_markdown(index=False, floatfmt=".2f"))
    elif keywords:
        res_lines.append(
            f"Geen specifieke transacties gevonden voor zoekwoorden: {', '.join(keywords)}."
        )

    # Categorie-aggregatie
    df_cat = pd.DataFrame()
    if found_categories and "Finny_GLDescription" in df_year.columns:
        cat_pattern = "|".join(re.escape(c) for c in found_categories)
        df_cat = df_year[
            df_year["Finny_GLDescription"]
            .astype(str)
            .str.contains(cat_pattern, case=False, na=False)
        ]

    if not df_cat.empty and "AmountDC_num" in df_cat.columns:
        pivot_cat = (
            df_cat.groupby("Year_Clean")["AmountDC_num"].sum().reset_index()
        )
        pivot_cat.columns = ["Jaar", f"Totaal categorie: {', '.join(found_categories)}"]
        res_lines.append("")
        res_lines.append(
            f"**Context: totale kosten binnen categorie(ën) {', '.join(found_categories)}:**"
        )
        res_lines.append(pivot_cat.to_markdown(index=False, floatfmt=".2f"))

        # Top 5 beschrijvingen binnen categorie
        top = (
            df_cat.groupby("Description")["AmountDC_num"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top.columns = ["Omschrijving", "Bedrag"]
        res_lines.append("")
        res_lines.append("*Top 5 grootste posten binnen de categorie:*")
        res_lines.append(top.to_markdown(index=False, floatfmt=".2f"))

    if len(res_lines) == 1:
        res_lines.append("Geen extra detail gevonden in de transacties.")

    return "\n\n".join(res_lines)


# ==========================================
# 6. HULPFUNCTIES VOOR LOGGING & NAVIGATIE
# ==========================================

def start_new_conversation() -> None:
    """Slaat huidig gesprek op en reset de chat."""
    if st.session_state.messages:
        # Titel op basis van eerste userbericht
        title = "Gesprek zonder titel"
        for m in st.session_state.messages:
            if m["role"] == "user":
                title = m["content"][:60]
                if len(m["content"]) > 60:
                    title += "..."
                break
        conv = {
            "id": datetime.now().isoformat(),
            "title": title,
            "timestamp": datetime.now(),
            "messages": st.session_state.messages.copy(),
            "shared_with_accountant": False,
        }
        st.session_state.conversations.append(conv)

    st.session_state.messages = []
    st.session_state.active_years = AVAILABLE_YEARS.copy()
    st.session_state.current_view = "chat"


def navigate_to(view_name: str) -> None:
    st.session_state.current_view = view_name


# ==========================================
# 7. HOOFDAPP
# ==========================================

if check_password():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Geen OPENAI_API_KEY gevonden in .env of st.secrets.")
        st.stop()

    client = OpenAI(api_key=api_key)
    data = load_data()

    # ---- SIDEBAR ----
    with st.sidebar:
        # Logo (neem eerste png/jpg in de map)
        logo_files = [f for f in os.listdir(".") if f.lower().endswith((".png", ".jpg"))]
        if logo_files:
            st.image(logo_files[0], width=150)
        st.title("Finny")

        # Geheugen + beschikbare jaren
        current_active = st.session_state.get("active_years", AVAILABLE_YEARS)
        st.caption(f"Geheugen (focusjaren): {current_active}")
        st.caption(f"Beschikbaar in deze demo: {AVAILABLE_YEARS}")

        st.markdown("---")

        if st.button("Nieuw gesprek", use_container_width=True):
            start_new_conversation()
            st.rerun()

        # Navigatie
        view_choice = st.radio(
            "Menu",
            ("chat", "intro", "history", "share"),
            format_func=lambda v: {
                "chat": "Chat",
                "intro": "Kennismaking",
                "history": "Eerdere gesprekken",
                "share": "Deel met accountant",
            }[v],
        )
        if view_choice != st.session_state.current_view:
            navigate_to(view_choice)
            st.rerun()

        # Overzicht gemarkeerde gesprekken
        shared_count = sum(
            1 for c in st.session_state.conversations if c.get("shared_with_accountant")
        )
        if shared_count:
            st.info(f"Gemarkeerd voor accountant: {shared_count} gesprekken")

    # ---- VIEW ROUTER ----
    view = st.session_state.current_view

    # ===== VIEW: CHAT =====
    if view == "chat":
        st.title("Finny Demo - Chat")

        # Toon chatgeschiedenis
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Nieuwe vraag
        if prompt := st.chat_input("Vraag Finny..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Finny denkt na..."):
                    intent = get_intent(client, prompt)

                    if intent["source"] == "PDF":
                        context = data.get("pdf_text", "")
                        st.caption(f"Bron: Jaarrekeningen | Jaren: {intent['years']}")
                    else:
                        context = analyze_csv_costs(data, intent)
                        st.caption(
                            f"Bron: Transacties | Jaren: {intent['years']} | Keywords: {intent['keywords']}"
                        )

                    system_prompt_finny = """
Je bent Finny, een informele maar scherpe financiële assistent voor mkb-ondernemers.

STIJL:
- Spreek de gebruiker aan met 'je' en 'jij'.
- Geen brievenstijl, geen aanhef of afsluitformules.
- Kort, duidelijk en eerlijk. Liever een "dat weet ik niet" dan verzonnen cijfers.

INSTRUCTIES:
- Je krijgt onder DATA de relevante stukken uit de jaarrekening (PDF) of een analyse van de transacties (CSV).
- Gebruik alleen getallen die je in DATA ziet. Verzin geen bedragen.
- Als DATA onvoldoende is om een harde uitspraak te doen, leg dat uit en geef eventueel suggesties welke extra informatie nodig is.
""".strip()

                    # Bouw messages payload
                    messages_payload = [
                        {
                            "role": "system",
                            "content": f"{system_prompt_finny}\n\nDATA:\n{context}",
                        }
                    ]
                    # Neem laatste 5 berichten uit de chat mee als context
                    for m in st.session_state.messages[-5:]:
                        role = "user" if m["role"] == "user" else "assistant"
                        messages_payload.append(
                            {"role": role, "content": m["content"]}
                        )

                    try:
                        res = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages_payload,
                        )
                        reply = res.choices[0].message.content
                    except Exception as e:  # pragma: no cover
                        reply = f"Er ging iets mis bij het ophalen van een antwoord: {e}"

                    st.markdown(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )

    # ===== VIEW: KENNISMAKING =====
    elif view == "intro":
        st.title("Kennismaking met Finny")
        st.write("Help Finny je beter te begrijpen door dit profiel in te vullen.")

        current_profile = st.session_state.client_profile or {}

        with st.form("intro_form"):
            st.subheader("Jouw kennisniveau")
            know_fin = st.slider(
                "Kennis van financiën (winst, omzet)",
                1,
                5,
                current_profile.get("finance_knowledge", 2),
            )
            know_tax = st.slider(
                "Kennis van belastingen (BTW, VPB)",
                1,
                5,
                current_profile.get("tax_knowledge", 2),
            )
            know_book = st.slider(
                "Kennis van boekhouden",
                1,
                5,
                current_profile.get("bookkeeping_knowledge", 1),
            )

            st.subheader("Voorkeuren")
            risk = st.select_slider(
                "Risico vs zekerheid",
                options=[1, 2, 3, 4, 5],
                value=current_profile.get("risk_preference", 3),
                format_func=lambda x: {
                    1: "1. Veiligheid eerst",
                    3: "3. Gemengd",
                    5: "5. Groeikansen pakken",
                }.get(x, str(x)),
            )

            focus = st.text_area(
                "Waar wil je dat Finny vooral bij helpt?",
                value=current_profile.get("focus_areas", ""),
            )
            avoid = st.text_area(
                "Onderwerpen die we over kunnen slaan?",
                value=current_profile.get("avoid_topics", ""),
            )

            if st.form_submit_button("Profiel opslaan"):
                st.session_state.client_profile = {
                    "finance_knowledge": know_fin,
                    "tax_knowledge": know_tax,
                    "bookkeeping_knowledge": know_book,
                    "risk_preference": risk,
                    "focus_areas": focus,
                    "avoid_topics": avoid,
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

        if st.button("Terug naar chat"):
            navigate_to("chat")
            st.rerun()

    # ===== VIEW: EERDERE GESPREKKEN =====
    elif view == "history":
        st.title("Eerdere gesprekken")
        st.write("Overzicht van opgeslagen sessies.")

        if not st.session_state.conversations:
            st.info(
                "Nog geen gesprekken opgeslagen. Gebruik 'Nieuw gesprek' om te starten."
            )
        else:
            for conv in sorted(
                st.session_state.conversations,
                key=lambda c: c["timestamp"],
                reverse=True,
            ):
                label = (
                    f"{conv['timestamp'].strftime('%d-%m %H:%M')} - {conv['title']}"
                )
                with st.expander(label):
                    st.caption(f"Berichten: {len(conv['messages'])}")
                    for m in conv["messages"]:
                        role = "Gebruiker" if m["role"] == "user" else "Finny"
                        st.markdown(f"**{role}:** {m['content']}")

        if st.button("Terug naar chat"):
            navigate_to("chat")
            st.rerun()

    # ===== VIEW: DEEL MET ACCOUNTANT =====
    elif view == "share":
        st.title("Deel met accountant")
        st.write(
            "Markeer de gesprekken die je wilt delen. In deze demo wordt er nog niets daadwerkelijk verstuurd."
        )

        if not st.session_state.conversations:
            st.info("Geen gesprekken om te delen.")
        else:
            with st.form("share_form"):
                selection: Dict[int, bool] = {}
                for idx, conv in enumerate(st.session_state.conversations):
                    label = (
                        f"{conv['timestamp'].strftime('%d-%m %H:%M')} - {conv['title']}"
                    )
                    checked = st.checkbox(
                        label,
                        value=conv.get("shared_with_accountant", False),
                        key=f"share_{idx}",
                    )
                    selection[idx] = checked

                    with st.expander("Bekijk inhoud"):
                        for m in conv["messages"]:
                            role = "Gebruiker" if m["role"] == "user" else "Finny"
                            st.write(f"{role}: {m['content'][:120]}")

                if st.form_submit_button("Bevestig selectie"):
                    for idx, checked in selection.items():
                        st.session_state.conversations[idx][
                            "shared_with_accountant"
                        ] = checked
                    st.success(
                        "De gemarkeerde gesprekken zijn aangemerkt om te delen met je accountant."
                    )
                    st.rerun()
