import streamlit as st
import pandas as pd
import re
from datetime import datetime

# --- 1. INITIALISATIE & STATE MANAGEMENT ---

def init_state():
    """Initialiseer alle sessie-variabelen."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hallo! Ik ben Finny. Vraag me alles over je cijfers (PDF jaarrekening of CSV transacties)."}
        ]
    
    # Cache voor harde berekeningen (Tuple key: (categorie, jaar) -> Bedrag/Context)
    if "metric_cache" not in st.session_state:
        st.session_state.metric_cache = {}
    
    # Context geheugen voor de conversatie
    if "active_years" not in st.session_state:
        st.session_state.active_years = [datetime.now().year - 1] # Default vorig jaar
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = None

# Roep init aan bij start
init_state()

# --- 2. DATA LADEN (Mockup - vervang dit met je echte file loaders) ---

@st.cache_data
def load_data():
    # Hier laad je normaliter je echte CSV's en PDF-teksten
    # Voor deze code ga ik uit van reeds geladen DataFrames in een echte app.
    # Dit is placeholders code.
    
    # df_trans = pd.read_csv("Finny_Transactions.csv")
    # df_syn = pd.read_csv("Finny_Synonyms.csv")
    # df_rgs = pd.read_csv("Finny_RGS.csv")
    
    # Even fake data om de logica te testen als je dit script draait zonder files
    data = {
        'Finny_Year': [2024]*100,
        'Finny_GLCode': [4001, 4001, 4002, 8000, 4001], # 4xxx = kosten
        'AmountDC_num': [50.00, 100.00, 2042.78, 50000.00, -10.00], # negatief kan correctie zijn
        'Description': ['Vodafone', 'Ziggo', 'KPN Zakelijk', 'Omzet', 'Credit Vodafone']
    }
    df_trans = pd.DataFrame(data)
    
    # Synonymen mapping
    df_syn = pd.DataFrame({
        'Keyword': ['telefoon', 'gsm', 'vodafone', 'ziggo', 'mobiel'],
        'Category': ['telefoonkosten', 'telefoonkosten', 'telefoonkosten', 'telefoonkosten', 'telefoonkosten']
    })
    
    # RGS / GL Mapping (Categorie -> GL Codes)
    # In het echt join je dit, hier simpel:
    gl_mapping = {
        'telefoonkosten': [4001, 4002],
        'communicatiekosten': [4001, 4002, 4005]
    }
    
    return df_trans, df_syn, gl_mapping

df_trans, df_syn, gl_mapping = load_data()

# --- 3. ROUTER & INTENT BEPALING ---

def extract_years(text):
    """Haal jaartallen uit de tekst. Update session_state als gevonden."""
    years = re.findall(r'\b(202[0-9])\b', text)
    if years:
        unique_years = sorted(list(set([int(y) for y in years])))
        st.session_state.active_years = unique_years
        return unique_years
    return st.session_state.active_years

def determine_intent(user_input):
    """
    Bepaalt: Bron (PDF/CSV), Categorie en Jaren.
    """
    user_input_lower = user_input.lower()
    years = extract_years(user_input)
    
    # 3.1 CSV Triggers (Harde lijst)
    csv_keywords = [
        "telefoon", "telefoonkosten", "bellen", "gsm", "vodafone", "ziggo", 
        "kpn", "communicatie", "transacties", "boekingen", "facturen", 
        "leverancier", "top 10", "details"
    ]
    
    # 3.2 PDF Triggers
    pdf_keywords = ["winst", "omzet", "totaal kosten", "resultaat", "jaarrekening", "ebitda"]
    
    intent = {
        "source": "PDF", # Default fallback
        "category": None,
        "years": years,
        "display_metric": "Algemeen"
    }

    # Logica: CSV wint als specifieke transactie-keywords worden genoemd
    if any(kw in user_input_lower for kw in csv_keywords):
        intent["source"] = "CSV"
        
        # Probeer categorie te vinden in synoniemen
        for idx, row in df_syn.iterrows():
            if row['Keyword'] in user_input_lower:
                intent["category"] = row['Category']
                break
        
        # Fallback als we wel CSV willen maar geen synoniem vinden (bv "transacties 2024")
        if not intent["category"]:
            intent["category"] = "general_transactions"

    elif any(kw in user_input_lower for kw in pdf_keywords):
        intent["source"] = "PDF"
        intent["display_metric"] = "Financieel overzicht"

    # 3.3 Intent Context Beware (Vraag: "En wat zijn de telefoonkosten?" -> pakt jaar uit memory)
    # Dit is al gedekt door st.session_state.active_years te gebruiken in 'extract_years'
    
    return intent

# --- 4. CALCULATION ENGINE (DETERMINISTISCH) ---

def calculate_csv_metrics(intent):
    """
    Voert de harde berekening uit op de DataFrame.
    Gebruikt caching om consistentie te garanderen.
    """
    category = intent["category"]
    years = intent["years"]
    
    if not category:
        return "Ik zie dat je details wilt, maar ik herken de specifieke kostensoort niet. Probeer 'telefoonkosten' of 'autokosten'."

    # Check Cache Eerst!
    cache_key = (category, tuple(years))
    if cache_key in st.session_state.metric_cache:
        return st.session_state.metric_cache[cache_key]

    # Zoek GL Codes
    target_gl_codes = gl_mapping.get(category, [])
    
    if not target_gl_codes and category != "general_transactions":
        return f"In je administratie (RGS) zie ik geen specifieke grootboekrekening voor '{category}'. Daardoor kan ik geen betrouwbaar totaal geven."

    # Filter Transacties
    mask_year = df_trans['Finny_Year'].isin(years)
    
    if category == "general_transactions":
        # Bij algemene vragen geen GL filter, maar wel oppassen met balansposten
        filtered_df = df_trans[mask_year]
    else:
        mask_gl = df_trans['Finny_GLCode'].isin(target_gl_codes)
        filtered_df = df_trans[mask_year & mask_gl]

    # Berekenen
    total_amount = filtered_df['AmountDC_num'].sum()
    count = len(filtered_df)
    
    # Format context string voor LLM
    year_str = ", ".join(map(str, years))
    
    # Context bouwen
    context_data = (
        f"CONTEXT DATA (Bron: CSV Transacties):\n"
        f"- Categorie: {category}\n"
        f"- Jaren: {year_str}\n"
        f"- Gebruikte GL Codes: {target_gl_codes}\n"
        f"- Aantal transacties: {count}\n"
        f"- Totaalbedrag: € {total_amount:,.2f} (Exact berekend)\n"
        f"INSTRUCTIE: Gebruik dit bedrag EXACT. Ga niet zelf rekenen of afronden."
    )
    
    if count == 0:
        context_data = f"CONTEXT: Geen transacties gevonden voor {category} in {year_str}."

    # Opslaan in cache
    st.session_state.metric_cache[cache_key] = context_data
    
    return context_data

def get_pdf_context(intent):
    """
    Simulatie van PDF retrieval.
    """
    year_str = ", ".join(map(str, intent["years"]))
    return f"CONTEXT DATA (Bron: PDF Jaarrekening):\n- Gevonden in jaarrekening {year_str}: De winst na belastingen is € 76.226."

# --- 5. LLM INTEGRATIE ---

def generate_response(user_input):
    # 1. Bepaal Intent
    intent = determine_intent(user_input)
    st.session_state.last_intent = intent

    # 2. Haal harde data op (Context)
    if intent["source"] == "CSV":
        context_text = calculate_csv_metrics(intent)
    else:
        context_text = get_pdf_context(intent)

    # 3. System Prompt
    system_prompt = f"""
    Je bent Finny, een financiële assistent.
    
    BELANGRIJKE REGELS:
    1. Je baseert je antwoord UITSLUITEND op de onderstaande CONTEXT.
    2. Als de context harde cijfers bevat (zoals € 2.192,78), neem je die EXACT over.
    3. Ga NOOIT zelf bedragen bij elkaar optellen, schatten of verzinnen.
    4. Als de context zegt "Geen transacties", zeg dan eerlijk dat je het niet weet. Ga geen andere categorieën (zoals autokosten) erbij halen als "telefoonkosten".
    5. Wees beknopt en direct.
    
    {context_text}
    """

    # 4. Call LLM (Hier mocken we de OpenAI call)
    # response = openai.ChatCompletion.create(...)
    
    # Mock response voor demo doeleinden:
    if "telefoon" in user_input.lower() and intent["source"] == "CSV":
        # We simuleren wat het LLM zou doen met de context_text
        # Extract het bedrag uit de context string voor de demo
        import re
        match = re.search(r'€ [\d,.]+', context_text)
        amount = match.group(0) if match else "onbekend"
        return f"Je telefoonkosten in {intent['years'][0]} bedragen {amount}. Dit is gebaseerd op de transacties die geboekt zijn onder communicatiekosten."
    
    elif intent["source"] == "PDF":
         return f"In {intent['years'][0]} bedraagt je winst na belastingen € 76.226 (volgens de jaarrekening)."
    
    else:
        return "Ik heb context geladen, maar kan geen specifiek antwoord formuleren in deze demo modus."

# --- 6. UI LOGICA ---

st.title("Finny Demo - Strakke CSV/PDF Router")

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Stel je financiële vraag..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Assistant logic
    response_text = generate_response(prompt)
    
    # Assistant message
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.write(response_text)
        
    # Debugging info tonen (optioneel, handig voor jou)
    with st.expander("Debug Info (Intent & Context)"):
        st.write(f"Intent: {st.session_state.last_intent}")
        if st.session_state.last_intent:
            cat = st.session_state.last_intent.get("category")
            yrs = tuple(st.session_state.last_intent.get("years"))
            if cat:
                st.write(f"Cached Data: {st.session_state.metric_cache.get((cat, yrs))}")
