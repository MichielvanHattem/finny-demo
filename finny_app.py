import streamlit as st
import pandas as pd
import re
import os
import glob

# Probeer PyPDF2 te gebruiken voor PDF-lezing, maar laat de app ook draaien zonder
try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:
    PdfReader = None  # type: ignore

# ------------------------------------------------------------------
# 1. CONFIGURATIE & SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Finny AI", page_icon="📊", layout="wide")

# Forceer dat kolommen die codes bevatten als tekst worden gelezen
DTYPE_SETTINGS = {
    "Finny_GLCode": str,
    "RGS_Referentiecode": str,
    "AccountCode_norm": str,
}

# ------------------------------------------------------------------
# 2. DATA LADEN (CSV's)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Laad Transacties
        df_trans = pd.read_csv("Finny_Transactions.csv", sep=";", dtype=DTYPE_SETTINGS)

        # Fix: zet komma-getallen om naar punt-getallen (floats)
        if "AmountDC_num" in df_trans.columns and df_trans["AmountDC_num"].dtype == object:
            df_trans["AmountDC_num"] = pd.to_numeric(
                df_trans["AmountDC_num"].astype(str).str.replace(",", "."),
                errors="coerce",
            )

        # Laad Synoniemen & RGS
        df_syn = pd.read_csv("Finny_Synonyms.csv", sep=";", dtype=DTYPE_SETTINGS)
        df_syn["Synoniem_lower"] = df_syn["Synoniem"].astype(str).str.lower()

        df_rgs = pd.read_csv("Finny_RGS.csv", sep=";", dtype=DTYPE_SETTINGS)

        return df_trans, df_syn, df_rgs
    except Exception as e:
        st.error(f"Fout bij laden CSV data: {e}")
        return None, None, None


df_trans, df_syn, df_rgs = load_data()

# ------------------------------------------------------------------
# 3. PDF LOGICA (JAARREKENING CHECK, MET PyPDF2)
# ------------------------------------------------------------------
def search_pdfs(query: str):
    """Zoekt simpel naar trefwoorden in PDF-bestanden (jaarrekeningen).
    Als PyPDF2 niet beschikbaar is of er zijn geen PDF's, geeft None terug.
    """
    if PdfReader is None:
        # PyPDF2 is niet geïnstalleerd; sla PDF-zoekfunctionaliteit over
        return None

    # Zoek alle PDF's in de map
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        return None

    query_words = query.lower().split()
    stop_words = ["de", "het", "een", "in", "van", "wat", "zijn", "kosten", "hoeveel", "finny"]
    search_terms = [w for w in query_words if w not in stop_words and len(w) > 3]

    if not search_terms:
        return None

    results = []

    for pdf_file in pdf_files:
        try:
            with open(pdf_file, "rb") as f:
                reader = PdfReader(f)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text() or ""
                    except Exception:
                        continue

                    text_lower = text.lower()
                    score = sum(1 for term in search_terms if term in text_lower)

                    if score > 0:
                        # Maak een leesbare snippet rondom de eerste match
                        snippet = "..."
                        for term in search_terms:
                            idx = text_lower.find(term)
                            if idx != -1:
                                start = max(0, idx - 50)
                                end = min(len(text), idx + 150)
                                snippet = text[start:end].replace("\n", " ")
                                break

                        results.append(
                            {
                                "file": os.path.basename(pdf_file),
                                "page": page_num + 1,
                                "snippet": snippet,
                                "score": score,
                            }
                        )
        except Exception:
            # Sla dit bestand over als het niet leesbaar is
            continue

    if not results:
        return None

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3]


# ------------------------------------------------------------------
# 4. CSV LOGICA (HET 3-STAPPEN PLAN)
# ------------------------------------------------------------------
def extract_years(question: str):
    """Haalt jaartallen uit de vraag."""
    years = re.findall(r"\b(20\d{2})\b", question)
    return [int(y) for y in years]


def find_gl_codes(question: str, df_syn: pd.DataFrame, df_rgs: pd.DataFrame):
    """Vertaalt vraag naar GL-codes via Synoniemen of RGS."""
    question_lower = question.lower()
    found_codes = set()
    debug_matches = []

    # 1A. Check Synoniemenlijst (prioriteit)
    for _, row in df_syn.iterrows():
        syn = row["Synoniem_lower"]
        if syn and syn in question_lower:
            found_codes.add(row["Finny_GLCode"])
            debug_matches.append(f"Synoniem '{row['Synoniem']}' -> GL {row['Finny_GLCode']}")

    # 1B. Back-up: zoek in RGS-omschrijvingen
    if not found_codes:
        for _, row in df_rgs.iterrows():
            omschrijving = str(row.get("RGS_Omschrijving", "")).lower()
            if omschrijving and omschrijving in question_lower:
                found_codes.add(row["Finny_GLCode"])
                debug_matches.append(
                    f"RGS match '{row.get('RGS_Omschrijving', '')}' -> GL {row['Finny_GLCode']}"
                )

    return list(found_codes), debug_matches


def get_financial_answer(question: str, df_trans: pd.DataFrame, gl_codes):
    """Filtert transacties en berekent totalen."""
    # Stap 2: Filter op Finny_GLCode
    filtered_df = df_trans[df_trans["Finny_GLCode"].isin(gl_codes)].copy()

    # Stap 2B: Filter op jaar (indien in vraag)
    years = extract_years(question)
    if years and "Finny_Year" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Finny_Year"].isin(years)]
        period_text = ", ".join(map_
