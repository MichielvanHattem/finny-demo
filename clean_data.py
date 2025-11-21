import pandas as pd
import os

def clean_dutch_number(val):
    """Zet € 1.234,56 om naar 1234.56 (float)"""
    if pd.isna(val) or val == '': return 0.0
    s = str(val).replace('€', '').replace(' ', '')
    s = s.replace('.', '') # Duizendtal weg
    s = s.replace(',', '.') # Komma naar punt
    try:
        return float(s)
    except:
        return 0.0

def run_cleanup():
    print("--- START DATA CLEANING ---")

    # 1. TRANSACTIES SCHOONMAKEN
    raw_trans_file = "133700 FinTransactionSearch all 5jr.csv"
    if os.path.exists(raw_trans_file):
        print(f"Verwerken van {raw_trans_file}...")
        try:
            # Lees ruwe data (Twinfield gebruikt vaak puntkomma)
            df = pd.read_csv(raw_trans_file, sep=';', dtype=str, on_bad_lines='skip')
            
            # Maak kolomnamen lowercase en strip spaties
            df.columns = df.columns.str.strip().str.lower()

            # Selecteer en hernoem alleen wat we nodig hebben
            # Pas deze mapping aan als de kolomnamen in jouw export anders zijn
            rename_map = {
                'reportingyear': 'jaar',
                'entrydate': 'datum',
                'glaccountcode': 'grootboek',
                'description': 'omschrijving',
                'amountdc': 'bedrag',
                'relationname': 'relatie',  # Indien aanwezig
                'dim1': 'grootboek'         # Soms heet het dim1
            }
            
            # Filter kolommen die bestaan in de CSV
            available_cols = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=available_cols)
            
            # Behoud alleen de nieuwe 'schone' kolommen
            target_cols = ['jaar', 'datum', 'grootboek', 'omschrijving', 'bedrag', 'relatie']
            existing_target_cols = [c for c in target_cols if c in df.columns]
            df = df[existing_target_cols]

            # Formatteer de data
            if 'bedrag' in df.columns:
                df['bedrag'] = df['bedrag'].apply(clean_dutch_number)
            
            if 'datum' in df.columns:
                # Forceer datum formaat YYYY-MM-DD
                df['datum'] = pd.to_datetime(df['datum'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')

            # Sla op als standaard CSV (komma gescheiden, punt voor decimalen)
            df.to_csv("finny_transacties_clean.csv", index=False, sep=",")
            print("✅ Succes: 'finny_transacties_clean.csv' aangemaakt.")
            
        except Exception as e:
            print(f"❌ Fout bij transacties: {e}")
    else:
        print(f"⚠️ Bestand niet gevonden: {raw_trans_file}")

    # 2. REKENINGSCHEMA SCHOONMAKEN
    raw_ledger_file = "133700 Standaard Rekeningschema Template FinGLAccountSearch.csv"
    if os.path.exists(raw_ledger_file):
        print(f"Verwerken van {raw_ledger_file}...")
        try:
            # Rekeningschema is vaak tab-separated in Twinfield exports
            df = pd.read_csv(raw_ledger_file, sep='\t', dtype=str, on_bad_lines='skip')
            df.columns = df.columns.str.strip().str.lower()

            rename_map = {
                'code': 'grootboek',
                'name': 'categorie_naam',
                'description': 'categorie_naam',
                'rgscode': 'rgs'
            }
            available_cols = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=available_cols)

            # Zorg dat grootboek codes matchen (soms voorloopnullen)
            if 'grootboek' in df.columns:
                 df['grootboek'] = df['grootboek'].str.replace(r'[^0-9]', '', regex=True) # Alleen cijfers

            df.to_csv("finny_rekeningschema_clean.csv", index=False, sep=",")
            print("✅ Succes: 'finny_rekeningschema_clean.csv' aangemaakt.")

        except Exception as e:
             print(f"❌ Fout bij rekeningschema: {e}")
    else:
        print(f"⚠️ Bestand niet gevonden: {raw_ledger_file}")

if __name__ == "__main__":
    run_cleanup()
