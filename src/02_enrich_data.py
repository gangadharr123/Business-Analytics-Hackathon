import pandas as pd
import holidays
import os

# --- PATH SETUP ---
# 1. Identify where this script is (src folder)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Identify the project root (one level up)
ROOT_DIR = os.path.dirname(SRC_DIR)
# 3. Define the Data Directory
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Inputs
INPUT_FILE = os.path.join(SRC_DIR, 'ebs_commute_data.csv') # Reads the file from 01
CONSTRUCTION_FILE = os.path.join(DATA_DIR, 'construction.csv') # <--- Reads from DATA folder
STRIKES_FILE = os.path.join(DATA_DIR, 'strikes.csv')           # <--- Reads from DATA folder

# Output (Save enriched data to DATA folder)
OUTPUT_FILE = os.path.join(DATA_DIR, 'ebs_commute_data_enriched.csv')

def process_construction_data(df_main):
    """Parses construction.csv and maps impacts to dates"""
    if not os.path.exists(CONSTRUCTION_FILE):
        print(f"⚠️ Warning: {CONSTRUCTION_FILE} not found. Skipping.")
        df_main['construction_impact'] = 0
        return df_main

    print(f"🚧 Reading Construction Data from {CONSTRUCTION_FILE}...")
    try:
        const_df = pd.read_csv(CONSTRUCTION_FILE)
        
        # 1. Map text impact to numbers
        impact_map = {'High': 3, 'Medium': 2, 'Low': 1}
        const_df['impact_score'] = const_df['impact_level'].map(impact_map).fillna(1).astype(int)
        
        # 2. Create a daily dictionary for lookup
        date_impact = {}
        for _, row in const_df.iterrows():
            try:
                dates = pd.date_range(start=row['start_date'], end=row['end_date'])
                score = row['impact_score']
                for d in dates:
                    d_date = d.date()
                    date_impact[d_date] = max(date_impact.get(d_date, 0), score)
            except:
                continue

        # 3. Map to main dataframe
        df_main['construction_impact'] = df_main['time'].dt.date.map(date_impact).fillna(0).astype(int)
        
    except Exception as e:
        print(f"❌ Error reading construction file: {e}")
        df_main['construction_impact'] = 0
        
    return df_main

def process_strike_data(df_main):
    """Parses strikes.csv and marks rail strikes"""
    if not os.path.exists(STRIKES_FILE):
        print(f"⚠️ Warning: {STRIKES_FILE} not found. Skipping.")
        df_main['strike_impact'] = 0
        return df_main

    print(f"🚩 Reading Strike Data from {STRIKES_FILE}...")
    try:
        strike_df = pd.read_csv(STRIKES_FILE)
        strike_dates = set()
        
        for _, row in strike_df.iterrows():
            try:
                # Filter for Rail/Train keywords
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
        print(f"❌ Error reading strike file: {e}")
        df_main['strike_impact'] = 0
        
    return df_main

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Run 01_filter_data.py first.")
        exit()

    print(f"⏳ Loading {INPUT_FILE}...")
    df_train = pd.read_csv(INPUT_FILE)
    df_train['time'] = pd.to_datetime(df_train['time'])
    
    # 1. Holidays
    print("🎉 Identifying Public Holidays...")
    unique_years = df_train['time'].dt.year.unique().tolist()
    de_holidays = holidays.Germany(subdiv='HE', years=unique_years)
    df_train['is_holiday'] = df_train['time'].dt.date.apply(lambda x: 1 if x in de_holidays else 0)

    # 2. Process Events
    df_train = process_construction_data(df_train)
    df_train = process_strike_data(df_train)

    # 3. Save to DATA folder
    df_train.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Success! Enriched data saved to:\n   {OUTPUT_FILE}")