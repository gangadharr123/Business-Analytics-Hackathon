"""
02_enrich_data.py
-----------------
BUSINESS ANALYTICS HACKATHON - DATA ENRICHMENT
Step 2: External Factors Integration

Description:
    Enhances the raw commuter data by integrating external "Business Factors":
    1. Construction Events (Riedbahn, Track repairs)
    2. Labor Strikes (GDL, Verdi)
    3. Public Holidays (Hessen)
    
    This creates the "Contextual Dataset" required for realistic ML predictions.

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
INPUT_FILE = os.path.join(SRC_DIR, 'ebs_commute_data.csv') # From Step 1
CONSTRUCTION_FILE = os.path.join(DATA_DIR, 'construction.csv')
STRIKES_FILE = os.path.join(DATA_DIR, 'strikes.csv')

# Output
OUTPUT_FILE = os.path.join(DATA_DIR, 'ebs_commute_data_enriched.csv')

def process_construction_data(df_main):
    """Maps construction date ranges to an impact score (1-3)."""
    if not os.path.exists(CONSTRUCTION_FILE):
        print(f"   ⚠️ WARNING: Construction file not found. Assuming 0 impact.")
        df_main['construction_impact'] = 0
        return df_main

    print(f"   🚧 Integrating Construction Data...")
    try:
        const_df = pd.read_csv(CONSTRUCTION_FILE)
        
        # Map text impact to numeric scale
        impact_map = {'High': 3, 'Medium': 2, 'Low': 1}
        const_df['impact_score'] = const_df['impact_level'].map(impact_map).fillna(1).astype(int)
        
        # Create Daily Lookup Dictionary: {Date: Max_Impact}
        date_impact = {}
        for _, row in const_df.iterrows():
            try:
                # Expand start/end dates into a daily range
                dates = pd.date_range(start=row['start_date'], end=row['end_date'])
                score = row['impact_score']
                for d in dates:
                    d_date = d.date()
                    date_impact[d_date] = max(date_impact.get(d_date, 0), score)
            except:
                continue

        # Map to main data
        df_main['construction_impact'] = df_main['time'].dt.date.map(date_impact).fillna(0).astype(int)
        
    except Exception as e:
        print(f"   ❌ Error processing construction: {e}")
        df_main['construction_impact'] = 0
        
    return df_main

def process_strike_data(df_main):
    """Identifies rail-related strikes."""
    if not os.path.exists(STRIKES_FILE):
        print(f"   ⚠️ WARNING: Strike file not found. Assuming 0 impact.")
        df_main['strike_impact'] = 0
        return df_main

    print(f"   🚩 Integrating Strike Data...")
    try:
        strike_df = pd.read_csv(STRIKES_FILE)
        strike_dates = set()
        
        for _, row in strike_df.iterrows():
            try:
                # Filter for Rail relevance (ignore purely local bus strikes)
                mode = str(row.get('transport_mode', '')).lower()
                union = str(row.get('union', '')).lower()
                is_rail = 'rail' in mode or 'train' in mode or 'gdl' in union or 'evg' in union
                
                if is_rail:
                    dates = pd.date_range(start=row['start_date'], end=row['end_date'])
                    for d in dates:
                        strike_dates.add(d.date())
            except:
                continue

        df_main['strike_impact'] = df_main['time'].dt.date.apply(lambda x: 1 if x in strike_dates else 0)
        
    except Exception as e:
        print(f"   ❌ Error processing strikes: {e}")
        df_main['strike_impact'] = 0
        
    return df_main

if __name__ == "__main__":
    print("🔄 PIPELINE START: Enrichment...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"   ❌ FATAL: {INPUT_FILE} missing. Run Step 1 first.")
        exit()

    df_train = pd.read_csv(INPUT_FILE)
    df_train['time'] = pd.to_datetime(df_train['time'])
    
    # 1. Holidays (Dynamic Calculation)
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