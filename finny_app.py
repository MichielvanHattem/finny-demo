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
# 0. SETUP & CONFIG
# ==========================================
# Probeer logo te vinden
logo_files = glob.glob("finny_logo.png") + glob.glob("*.png") + glob.glob("*.jpg")
main_logo = "finny_logo.png" if os.path.exists("finny_logo.png") else (logo_files[0] if logo_files else "💰")

st.set_page_config(page_title="Finny Demo", page_icon=main_logo, layout="wide")
load_dotenv()

# Instellingen
AVAILABLE_YEARS = [2022, 2023, 2024]
CURRENT_YEAR_CAP = 2024  # Harde limiet om '2026' fouten te voorkomen

# --- STATE MANAGEMENT ---
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
                first_q = m["content"][:60] + ("..." if len(m["content"]) > 60 else "")
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

    # A. Transacties
    if os.path.exists("Finny_Transactions.csv"):
        try:
            df = pd.read_csv("Finny_Transactions.csv", sep=";", encoding="latin1")
            
            # Kolommen normaliseren (strip spaties in headers)
            df.columns = df.columns.str.strip()
            
            if "Finny_Year" in df.columns:
                df["Year_Clean"] = df["Finny_Year"].astype(str).str.split(".").str[0]
                valid_years = pd.to_numeric(df["Year_Clean"], errors="coerce").dropna()
                
                # FIX: Voorkom toekomstige jaren door CSV fouten
                if not valid_years.empty:
                    max_csv_year = int(valid_years.max())
                    data["latest_year"] = min(max_csv_year, CURRENT_YEAR_CAP)
            
            # Zorg dat de essentiële kolommen bestaan
            expected_cols = ["Description", "AccountName", "Finny_GLDescription", "Finny_GLCode"]
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""

            # Universal Search kolom maken
            df["UniversalSearch"] = df[expected_cols].astype(str).agg(" ".join, axis=1).str.lower()
            data["trans"] = df
        except Exception as e:
            st.error(f"Fout CSV: {e}")

    # B. Synoniemen
    if os.path.exists("Finny_Synonyms.csv"):
        try:
            df = pd.read_csv("Finny_Synonyms.csv", sep=";", dtype=str)
            if "Synoniem" in df.columns:
                df["Synoniem_Clean"] = df["Synoniem"].astype(str).str.lower().str.strip()
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
                data["company_context"] += f"\n--- PROFIEL ({txt}) ---\n{f.read()}\n"
        except Exception:
            try:
                with open(txt, "r", encoding="latin1") as f:
                    data["company_context"] += f"\n--- PROFIEL ({txt}) ---\n{f.read()}\n"
            except Exception:
                pass

    return data

# ==========================================
# 3. SLIMME LOGICA (ROUTER & ANALYSE)
# ==========================================

# FIX: Hardcoded mapping voor als de CSV synoniemen niet compleet zijn.
FALLBACK_MAPPING = {
    "telefoon": ["communicatie", "telefoon", "mobiel", "kpn", "vodafone", "t-mobile", "odido", "kantoor"],
    "mobiel": ["communicatie", "telefoon", "mobiel"],
    "internet": ["communicatie", "internet", "kpn", "ziggo", "kantoor"],
    "auto": ["auto", "brandstof", "tank", "lease", "onderhoud", "parkeren", "vervoer"],
    "vervoer": ["auto", "brandstof", "tank", "lease", "onderhoud", "parkeren", "vervoer", "trein", "ns"],
    "huisvesting": ["huur", "gas", "water", "licht", "energie", "pand", "huisvesting"],
}

def get_intent(client, question, data):
    """
    Slimme router die Context, Scope en Zoekwoorden bepaalt.
    """
    q_lower = question.lower()
    context_years = st.session_state.get("active_years", AVAILABLE_YEARS)

    # 1. Check Synoniemen (CSV + Fallback)
    csv_keywords = []
    
    # Eerst checken in hardcoded fallback (snelste fix voor 'telefoon')
    for key, values in FALLBACK_MAPPING.items():
        if key in q_lower:
            csv_keywords.extend(values)
    
    # Dan checken in CSV
    if data["syn"] is not None and not data["syn"].empty and "Synoniem_Clean" in data["syn"].columns:
        known_synonyms = data["syn"]["Synoniem_Clean"].dropna().unique()
        for syn in known_synonyms:
            if syn in q_lower and len(syn) > 2:
                csv_keywords.append(syn)
    
    # Unieke keywords maken
    csv_keywords = list(set(csv_keywords))

    # Als we keywords hebben, en het is geen trendvraag -> CSV Specific
    if csv_keywords:
        if not any(x in q_lower for x in ["waarom", "hoe komt", "oorzaak", "verloop", "vergelijk"]):
            return {
                "source": "CSV",
                "scope": "specific",
                "keywords": csv_keywords,
                "years": context_years,
                "needs_year": True,
                "year_mode": "single",
            }

    # 2. Check PDF signalen (Trends)
    TREND_TERMS = ["stijgen", "dalen", "toenemen", "afnemen", "waarom", "hoe komt", "oorzaak", "verloop", "trend", "verschil", "vergelijk", "ontwikkeling"]
    if any(t in q_lower for t in TREND_TERMS):
        return {
            "source": "PDF",
            "scope": "general",
            "keywords": [],
            "years": context_years,
            "needs_year": False, 
            "year_mode": "multi",
        }

    # 3. LLM Router (voor complexere vragen)
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
Je bent de router van Finny. Analyseer de vraag.

OUTPUT JSON:
1. "source": 'CSV' (bedragen/kosten/details) of 'PDF' (trends/uitleg).
2. "scope": 
   - 'specific' (vraag naar specifieke post, bv "telefoon", "huur").
   - 'total' (vraag naar totaaloverzicht, bv "wat zijn mijn kosten", "uitgaven").
   - 'general' (uitleg/hallo).
3. "keywords": Lijst met NL zoektermen als scope='specific'.
   - BELANGRIJK: Als gebruiker vraagt om "telefoon", geef keywords ["telefoon", "communicatie", "mobiel", "kantoor"].
4. "needs_year": true/false.
5. "year_mode": 'single' | 'multi' | 'none'.
6. "years": [2024] indien expliciet genoemd.
"""
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        intent = json.loads(res.choices[0].message.content)
        
        # Fallbacks
        if not intent.get("years"): intent["years"] = context_years
        if "needs_year" not in intent: intent["needs_year"] = False
        if "scope" not in intent: intent["scope"] = "general"
        
        # Merge LLM keywords met eventuele fallback keywords
        if intent.get("scope") == "specific":
            found_kws = intent.get("keywords", [])
            for k in found_kws:
                for map_key, map_vals in FALLBACK_MAPPING.items():
                    if map_key in k.lower():
                        found_kws.extend(map_vals)
            intent["keywords"] = list(set(found_kws))

        return intent
    except Exception:
        return {"source": "PDF", "scope": "general", "keywords": [], "years": context_years, "needs_year": False, "year_mode": "none"}

def analyze_csv_costs(data, intent):
    if data["trans"] is None:
        return "Geen transacties beschikbaar."
    
    df = data["trans"].copy()
    keywords = intent.get("keywords", [])
    scope = intent.get("scope", "general")
    years = [str(y) for y in intent.get("years", AVAILABLE_YEARS)]

    # 1. Filter op jaar
    if "Year_Clean" in df.columns:
        df = df[df["Year_Clean"].astype(str).isin(years)]
    
    # 2. Filter technische boekingen weg (Winst/Verlies/Balans)
    if "Description" in df.columns:
        df = df[~df["Description"].astype(str).str.contains(r"(resultaat|winst|balans|privé)", case=False, na=False)]

    df_result = pd.DataFrame()

    # 3. Filter op Scope
    if scope == "specific" and keywords:
        pat = "|".join([re.escape(k.lower()) for k in keywords])
        
        # A. Zoek in de Categorie Naam (Finny_GLDescription) - DIT IS HET BELANGRIJKSTE
        if "Finny_GLDescription" in df.columns:
            df_cat_match = df[df["Finny_GLDescription"].astype(str).str.contains(pat, case=False, na=False)]
        else:
            df_cat_match = pd.DataFrame()

        # B. Zoek in de Universele zoektekst (Description + GL + etc)
        if "UniversalSearch" in df.columns:
            df_uni_match = df[df["UniversalSearch"].astype(str).str.contains(pat, na=False)]
        else:
            df_uni_match = pd.DataFrame()

        df_result = pd.concat([df_cat_match, df_uni_match]).drop_duplicates()
        
    else:
        # Scope = Total: Pak alles
        df_result = df.copy()

    if df_result.empty:
        return f"Geen kosten gevonden voor '{', '.join(keywords)}' in {', '.join(years)}."

    # 4. Resultaat formatteren (Slimmer groeperen)
    total = df_result["AmountDC_num"].sum() if "AmountDC_num" in df_result.columns else 0
    
    res = f"### DETAILS ({', '.join(years)})\n"
    res += f"**Totaal geselecteerd:** € {total:,.2f}\n\n"

    # Groepeer per Categorie (GLDescription) ipv per losse factuurregel
    if "Finny_GLDescription" in df_result.columns and "AmountDC_num" in df_result.columns:
        res += "**Per categorie:**\n"
        grouped = df_result.groupby("Finny_GLDescription")["AmountDC_num"].sum().sort_values(ascending=False).reset_index()
        grouped = grouped[grouped["AmountDC_num"] != 0] # Lege categorieën weg
        res += grouped.to_markdown(index=False, floatfmt=".2f")
        res += "\n\n"
        
        # Als het een specifieke zoekopdracht was, toon dan ook de top 5 details (omschrijvingen)
        if scope == "specific":
            res += "**Top 5 boekingen:**\n"
            top_desc = df_result.groupby("Description")["AmountDC_num"].sum().sort_values(ascending=False).head(5).reset_index()
            res += top_desc.to_markdown(index=False, floatfmt=".2f")

    return res

# ==========================================
# 4. MAIN UI
# ==========================================
if check_password():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key: st.stop()
    client = OpenAI(api_key=api_key)
    data = load_data()

    # SIDEBAR
    with st.sidebar:
        if os.path.exists(main_logo): st.image(main_logo, width=150)
        st.title("Finny Demo")
        st.caption(f"Geheugen: {st.session_state.get('active_years', AVAILABLE_YEARS)}")
        st.markdown("---")
        if st.button("Nieuw Gesprek", use_container_width=True):
            start_new_conversation()
            st.rerun()
        # Menu
        opts = ["intro", "chat", "history", "share"]
        curr_idx = opts.index(st.session_state.current_view) if st.session_state.current_view in opts else 0
        choice = st.radio("Menu", opts, index=curr_idx, format_func=lambda x: {"chat": "Chat", "intro": "Start", "history": "Gesprekken", "share": "Accountant"}[x])
        if choice != st.session_state.current_view:
            st.session_state.current_view = choice
            st.rerun()

    view = st.session_state.current_view

    # --- VIEW: INTRO ---
    if view == "intro":
        st.title("Welkom bij Finny")
        st.write("Laten we even kennismaken.")
        curr = st.session_state.client_profile
        with st.form("prof"):
            fk = st.slider("Financiële kennis", 1, 5, curr.get("finance_knowledge", 2))
            if st.form_submit_button("Start Chat"):
                curr["finance_knowledge"] = fk
                st.session_state.client_profile = curr
                st.session_state.current_view = "chat"
                st.rerun()

    # --- VIEW: CHAT ---
    elif view == "chat":
        st.title("Finny Demo")
        finny_av = main_logo if os.path.exists(main_logo) else "🤖"
        user_av = st.session_state.client_profile.get("avatar") or "👤"

        for m in st.session_state.messages:
            st.chat_message(m["role"], avatar=finny_av if m["role"] == "assistant" else user_av).write(m["content"])

        if prompt := st.chat_input("Vraag Finny..."):
            
            # --- CONTEXT FIX: MERGE MET VORIGE VRAAG INDIEN NODIG ---
            processed_prompt = prompt
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                last_msg = st.session_state.messages[-1]["content"].lower()
                # Als Finny net een vraag stelde om verduidelijking...
                if "welk jaar" in last_msg or "welke periode" in last_msg or "welke kosten" in last_msg:
                    if len(st.session_state.messages) >= 2:
                        original_user_q = st.session_state.messages[-2]["content"]
                        # Plak de oude vraag aan het nieuwe antwoord vast voor de router
                        processed_prompt = f"{original_user_q} | Detail: {prompt}"

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user", avatar=user_av).write(prompt)
            
            with st.chat_message("assistant", avatar=finny_av):
                with st.spinner("..."):
                    # 1. Intent Bepalen (op basis van merged prompt)
                    intent = get_intent(client, processed_prompt, data)
                    source = intent.get("source", "PDF")
                    scope = intent.get("scope", "general")
                    
                    # 2. Expliciete jaren (Regex op originele prompt + merged)
                    year_matches = re.findall(r"\b(20[0-9]{2})\b", processed_prompt)
                    explicit_years = [int(y) for y in year_matches if int(y) in AVAILABLE_YEARS]
                    if explicit_years:
                        intent["years"] = explicit_years

                    # 3. Validatie: Mis ik iets?
                    user_years = intent.get("years", [])
                    needs_year = bool(intent.get("needs_year", False))
                    year_mode = intent.get("year_mode", "none")

                    # Mis ik een jaar? (Alleen als needs_year=True EN single mode EN geen expliciet jaar)
                    missing_year = needs_year and (year_mode == "single") and (not explicit_years)
                    
                    # Mis ik een categorie? (Alleen als source=CSV EN scope=Specific EN geen keywords)
                    missing_cat = False
                    if source == "CSV" and scope == "specific" and not intent.get("keywords"):
                        missing_cat = True

                    latest_year = data.get("latest_year", 2024)
                    
                    # 4. Actie: Vraag terugstellen OF Antwoorden
                    if missing_year or missing_cat:
                        reply_lines = ["Ik heb nog een klein detail nodig:"]
                        if missing_year and missing_cat:
                            reply_lines.append("Over welk jaar en welke kosten gaat het?")
                        elif missing_year:
                            reply_lines.append(f"Over welk jaar wil je dit weten? (Beschikbaar t/m {latest_year})")
                        elif missing_cat:
                            reply_lines.append("Over welke specifieke kosten gaat je vraag?")
                        
                        reply = "\n".join(reply_lines)
                        st.write(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    else:
                        # We hebben genoeg info -> Data ophalen
                        context = ""
                        caption_txt = ""
                        
                        if source == "CSV":
                            context = analyze_csv_costs(data, intent)
                            caption_txt = f"Bron: CSV Transacties | Jaren: {intent.get('years')} | Focus: {scope}"
                        else:
                            context = data["pdf_text"]
                            caption_txt = f"Bron: Jaarrekening Tekst | Jaren: {intent.get('years')}"

                        if context:
                            st.caption(caption_txt)
                            company_info = data["company_context"] or "Geen profiel."
                            
                            system_prompt = f"""
                            Je bent Finny.

                            BEDRIJF:
                            {company_info}

                            DATA:
                            {context}

                            OPDRACHT:
                            - Beantwoord de vraag met de DATA.
                            - Gebruik correcte termen: "Communicatiekosten" ipv "Communicatiemaatschappij".
                            - Geef bedragen en details.
                            - Max 4 zinnen.
                            """
                            
                            msgs = [{"role": "system", "content": system_prompt}]
                            for m in st.session_state.messages[-3:]:
                                msgs.append({"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]})
                            
                            res = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
                            reply = res.choices[0].message.content
                            st.write(reply)
                            st.session_state.messages.append({"role": "assistant", "content": reply})

    # --- VIEWS HISTORY & SHARE ---
    elif view == "history":
        st.title("Gesprekken")
        for c in reversed(st.session_state.conversations):
            with st.expander(f"{c['timestamp'].strftime('%d-%m %H:%M')} - {c['title']}"):
                for m in c["messages"]: 
                    st.write(f"**{m['role']}**: {m['content']}")

    elif view == "share":
        st.title("Accountant")
        st.write("Markeer gesprekken die je met je accountant wilt delen.")
        if not st.session_state.conversations:
            st.info("Er zijn nog geen gesprekken opgeslagen.")
        else:
            with st.form("share_form"):
                sel = {}
                for i, c in enumerate(st.session_state.conversations):
                    is_shared = c.get("shared", False)
                    sel[i] = st.checkbox(
                        f"{c['timestamp'].strftime('%d-%m %H:%M')} - {c['title']}",
                        value=is_shared,
                        key=f"share_{i}"
                    )
                st.markdown("---")
                if st.form_submit_button("Wijzigingen Opslaan"):
                    for i, checked in sel.items():
                        st.session_state.conversations[i]["shared"] = checked
                    st.success("De selectie is bijgewerkt!")
                    st.rerun()
