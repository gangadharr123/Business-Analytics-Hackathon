"""
02_enrich_data.py
-----------------
BUSINESS ANALYTICS HACKATHON - DATA ENRICHMENT
Step 2: External Factors Integration

Description:
    Enhances the raw commuter data by integrating external "Business Factors".
    Now includes a "Smart Reader" to handle German Excel (;) vs Standard CSV (,).

Author: Gangadhar
"""

import pandas as pd
import holidays
import os

# --- 1. PATH SETUP ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Inputs
INPUT_FILE = os.path.join(SRC_DIR, 'ebs_commute_data.csv')
CONSTRUCTION_FILE = os.path.join(DATA_DIR, 'construction.csv')
STRIKES_FILE = os.path.join(DATA_DIR, 'strikes.csv')

# Output
OUTPUT_FILE = os.path.join(DATA_DIR, 'ebs_commute_data_enriched.csv')

def read_csv_smart(file_path):
    """
    Smartly detects if a CSV is comma (,) or semicolon (;) separated.
    Also handles encoding (UTF-8 vs Latin1).
    """
    if not os.path.exists(file_path):
        return None

    filename = os.path.basename(file_path)
    encodings = ['utf-8', 'latin1', 'cp1252']
    separators = [',', ';'] # Try comma first, then semicolon
    
    for enc in encodings:
        for sep in separators:
            try:
                # Read just the first line to sniff the format
                df_peek = pd.read_csv(file_path, sep=sep, encoding=enc, nrows=2)
                
                # If we found more than 1 column, this is likely the correct separator!
                if len(df_peek.columns) > 1:
                    # print(f"   ...Detected format: Sep='{sep}' Encoding='{enc}' for {filename}")
                    return pd.read_csv(file_path, sep=sep, encoding=enc, on_bad_lines='skip')
            except:
                continue
                
    # Fallback: Python engine (slower but more forgiving)
    print(f"   ⚠️ Could not auto-detect format for {filename}. Trying fallback...")
    try:
        return pd.read_csv(file_path, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
    except Exception as e:
        print(f"   ❌ Failed to read {filename}: {e}")
        return pd.DataFrame()

def process_construction_data(df_main):
    """Maps construction date ranges to an impact score (1-3)."""
    if not os.path.exists(CONSTRUCTION_FILE):
        print(f"   ⚠️ WARNING: Construction file not found. Assuming 0 impact.")
        df_main['construction_impact'] = 0
        return df_main

    print(f"   🚧 Integrating Construction Data...")
    const_df = read_csv_smart(CONSTRUCTION_FILE)
    
    if const_df is None or const_df.empty:
        df_main['construction_impact'] = 0
        return df_main

    try:
        # Standardize column names (lowercase) to avoid 'Start_Date' vs 'start_date' issues
        const_df.columns = const_df.columns.str.lower().str.strip()
        
        # Map text impact to numeric scale
        impact_map = {'high': 3, 'medium': 2, 'low': 1}
        # Safely map impact level
        if 'impact_level' in const_df.columns:
            const_df['impact_score'] = const_df['impact_level'].astype(str).str.lower().map(impact_map).fillna(1).astype(int)
        else:
            const_df['impact_score'] = 1 # Default if column missing

        # Create Daily Lookup
        date_impact = {}
        for _, row in const_df.iterrows():
            try:
                if pd.notna(row['start_date']) and pd.notna(row['end_date']):
                    dates = pd.date_range(start=row['start_date'], end=row['end_date'])
                    score = row['impact_score']
                    for d in dates:
                        d_date = d.date()
                        date_impact[d_date] = max(date_impact.get(d_date, 0), score)
            except:
                continue

        df_main['construction_impact'] = df_main['time'].dt.date.map(date_impact).fillna(0).astype(int)
        
    except Exception as e:
        print(f"   ❌ Error logic in construction: {e}")
        df_main['construction_impact'] = 0
        
    return df_main

def process_strike_data(df_main):
    """Identifies rail-related strikes."""
    if not os.path.exists(STRIKES_FILE):
        print(f"   ⚠️ WARNING: Strike file not found. Assuming 0 impact.")
        df_main['strike_impact'] = 0
        return df_main

    print(f"   🚩 Integrating Strike Data...")
    strike_df = read_csv_smart(STRIKES_FILE)
    
    if strike_df is None or strike_df.empty:
        df_main['strike_impact'] = 0
        return df_main

    try:
        strike_df.columns = strike_df.columns.str.lower().str.strip()
        strike_dates = set()
        
        for _, row in strike_df.iterrows():
            try:
                # Filter for Rail relevance
                mode = str(row.get('transport_mode', '')).lower()
                union = str(row.get('union', '')).lower()
                is_rail = 'rail' in mode or 'train' in mode or 'gdl' in union or 'evg' in union
                
                if is_rail and pd.notna(row['start_date']) and pd.notna(row['end_date']):
                    dates = pd.date_range(start=row['start_date'], end=row['end_date'])
                    for d in dates:
                        strike_dates.add(d.date())
            except:
                continue

        df_main['strike_impact'] = df_main['time'].dt.date.apply(lambda x: 1 if x in strike_dates else 0)
        
    except Exception as e:
        print(f"   ❌ Error logic in strikes: {e}")
        df_main['strike_impact'] = 0
        
    return df_main

if __name__ == "__main__":
    print("🔄 PIPELINE START: Enrichment...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"   ❌ FATAL: {INPUT_FILE} missing. Run Step 1 first.")
        exit()

    df_train = pd.read_csv(INPUT_FILE)
    df_train['time'] = pd.to_datetime(df_train['time'])
    
    # 1. Holidays
    print("   🎉 Identifying Public Holidays (Hessen)...")
    unique_years = df_train['time'].dt.year.unique().tolist()
    de_holidays = holidays.Germany(subdiv='HE', years=unique_years)
    df_train['is_holiday'] = df_train['time'].dt.date.apply(lambda x: 1 if x in de_holidays else 0)

    # 2. External CSVs
    df_train = process_construction_data(df_train)
    df_train = process_strike_data(df_train)

    # 3. Save
    df_train.to_csv(OUTPUT_FILE, index=False)
    print(f"   ✅ SUCCESS: Enriched data saved to {OUTPUT_FILE}")