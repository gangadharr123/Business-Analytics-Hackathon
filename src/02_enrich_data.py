"""
02_enrich_data.py
-----------------
BUSINESS ANALYTICS HACKATHON - DATA ENRICHMENT
Step 2: External Factors Integration (Debugged)

Description:
    Integrates Weather, Construction, and Strikes.
    Debugs column names to fix 'impact_level' error.
    Uses flexible CSV engine to fix 'Expected 2 fields' error.

Author: Gangadhar
"""

import pandas as pd
import holidays
import os

# --- 1. PATH SETUP ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

INPUT_FILE = os.path.join(SRC_DIR, 'ebs_commute_data.csv')
CONSTRUCTION_FILE = os.path.join(DATA_DIR, 'construction.csv')
STRIKES_FILE = os.path.join(DATA_DIR, 'strikes.csv')
WEATHER_FILE = os.path.join(DATA_DIR, 'weather.csv')

OUTPUT_FILE = os.path.join(DATA_DIR, 'ebs_commute_data_enriched.csv')

def read_csv_bulletproof(file_path):
    """
    Tries to read messy CSVs using the flexible Python engine.
    """
    if not os.path.exists(file_path):
        return None

    # Try common separators with the python engine (more robust)
    separators = [',', ';', '\t']
    encodings = ['utf-8', 'latin1', 'cp1252']

    for enc in encodings:
        for sep in separators:
            try:
                # on_bad_lines='skip' ignores rows with too many commas
                df = pd.read_csv(file_path, sep=sep, encoding=enc, engine='python', on_bad_lines='skip')
                if len(df.columns) > 1:
                    return df
            except:
                continue
            
    print(f"   ❌ Failed to read {os.path.basename(file_path)} with any config.")
    return pd.DataFrame()

def process_weather_data(df_main):
    """Ingests ALL weather columns, handling metadata rows dynamically."""
    if not os.path.exists(WEATHER_FILE):
        print(f"   ⚠️ WARNING: {WEATHER_FILE} not found. Skipping weather.")
        return df_main

    print(f"   ☁️  Integrating FULL Weather Suite...")
    try:
        # 1. Find the Header Row
        skip_rows = 0
        with open(WEATHER_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if line.startswith('time') or 'time,' in line or 'time;' in line:
                    skip_rows = i
                    break
        
        # 2. Read Data
        try:
            w_df = pd.read_csv(WEATHER_FILE, skiprows=skip_rows, sep=',')
            if 'time' not in w_df.columns:
                 w_df = pd.read_csv(WEATHER_FILE, skiprows=skip_rows, sep=';')
        except:
            w_df = pd.read_csv(WEATHER_FILE, skiprows=skip_rows, sep=None, engine='python')

        # 3. Clean Columns
        w_df.columns = w_df.columns.str.strip().str.lower()
        
        if 'time' not in w_df.columns:
            return df_main

        # 4. Map ALL columns
        rename_map = {}
        for col in w_df.columns:
            if 'time' in col: rename_map[col] = 'time_key'
            elif 'temperature_2m' in col: rename_map[col] = 'temp_c'
            elif 'relative_humidity' in col: rename_map[col] = 'humidity_pct'
            elif 'precipitation' in col: rename_map[col] = 'precip_mm'
            elif 'rain' in col: rename_map[col] = 'rain_mm'
            elif 'snowfall' in col: rename_map[col] = 'snow_cm'
            elif 'apparent_temperature' in col: rename_map[col] = 'feels_like_c'
            elif 'snow_depth' in col: rename_map[col] = 'snow_depth_m'
            elif 'gusts' in col: rename_map[col] = 'wind_gusts_kmh'
            elif 'wind_speed_10m' in col: rename_map[col] = 'wind_speed_kmh'
            elif 'wind_speed_100m' in col: rename_map[col] = 'wind_speed_100m_kmh'
            elif 'direction_10m' in col: rename_map[col] = 'wind_dir_10m'
            elif 'direction_100m' in col: rename_map[col] = 'wind_dir_100m'

        w_df.rename(columns=rename_map, inplace=True)
        
        # 5. Create Merge Keys
        w_df['time_key'] = pd.to_datetime(w_df['time_key'])
        w_df['date'] = w_df['time_key'].dt.date
        w_df['hour'] = w_df['time_key'].dt.hour
        
        # 6. Select Columns to Keep
        keep_cols = list(rename_map.values())
        if 'time_key' in keep_cols: keep_cols.remove('time_key')
        final_cols = ['date', 'hour'] + [c for c in keep_cols if c in w_df.columns]
        
        w_df_clean = w_df[final_cols].copy()
        
        # 7. Merge
        df_main['time_temp'] = pd.to_datetime(df_main['time'])
        df_main['date'] = df_main['time_temp'].dt.date
        df_main['hour'] = df_main['time_temp'].dt.hour
        
        df_merged = pd.merge(df_main, w_df_clean, on=['date', 'hour'], how='left')
        
        fill_cols = [c for c in final_cols if c not in ['date', 'hour']]
        df_merged[fill_cols] = df_merged[fill_cols].fillna(0)
        
        df_merged.drop(columns=['time_temp', 'date'], inplace=True)
        return df_merged

    except Exception as e:
        print(f"   ❌ Error processing weather: {e}")
        return df_main

def process_construction_data(df_main):
    if not os.path.exists(CONSTRUCTION_FILE): return df_main
    print(f"   🚧 Integrating Construction Data...")
    
    const_df = read_csv_bulletproof(CONSTRUCTION_FILE)
    if const_df is None or const_df.empty: return df_main
    
    try:
        # Debugging: Print columns found
        const_df.columns = const_df.columns.str.lower().str.strip()
        # print(f"      (Found columns: {list(const_df.columns)})")
        
        impact_map = {'high': 3, 'medium': 2, 'low': 1}
        
        # Smart Column Search: Look for 'impact' or 'level' or 'severity'
        col_impact = 'impact_level' # Default
        for c in const_df.columns:
            if 'impact' in c or 'level' in c or 'severity' in c:
                col_impact = c
                break
        
        const_df['score'] = const_df[col_impact].astype(str).str.lower().map(impact_map).fillna(1)
        
        date_impact = {}
        for _, row in const_df.iterrows():
            try:
                # Ensure dates are valid
                if pd.notna(row['start_date']) and pd.notna(row['end_date']):
                    dates = pd.date_range(row['start_date'], row['end_date'])
                    for d in dates: 
                        date_impact[d.date()] = max(date_impact.get(d.date(),0), row['score'])
            except: continue
            
        df_main['construction_impact'] = df_main['time'].dt.date.map(date_impact).fillna(0).astype(int)
    except Exception as e: 
        print(f"   ⚠️ Construction logic error: {e}")
        df_main['construction_impact'] = 0
    return df_main

def process_strike_data(df_main):
    if not os.path.exists(STRIKES_FILE): return df_main
    print(f"   🚩 Integrating Strike Data...")
    
    strike_df = read_csv_bulletproof(STRIKES_FILE)
    if strike_df is None or strike_df.empty: return df_main
    
    try:
        strike_df.columns = strike_df.columns.str.lower().str.strip()
        strike_dates = set()
        
        for _, row in strike_df.iterrows():
            # Robust check for rail-related keywords in the entire row
            row_text = " ".join([str(val) for val in row.values]).lower()
            
            if 'rail' in row_text or 'train' in row_text or 'gdl' in row_text or 'evg' in row_text:
                try:
                    if pd.notna(row['start_date']) and pd.notna(row['end_date']):
                        dates = pd.date_range(row['start_date'], row['end_date'])
                        for d in dates: strike_dates.add(d.date())
                except: continue
        
        df_main['strike_impact'] = df_main['time'].dt.date.apply(lambda x: 1 if x in strike_dates else 0)
    except Exception as e:
        print(f"   ⚠️ Strike logic error: {e}")
        df_main['strike_impact'] = 0
    return df_main

if __name__ == "__main__":
    print("🔄 PIPELINE START: Enrichment...")
    if not os.path.exists(INPUT_FILE):
        print(f"   ❌ FATAL: {INPUT_FILE} missing.")
        exit()

    df_train = pd.read_csv(INPUT_FILE)
    df_train['time'] = pd.to_datetime(df_train['time'])
    
    # Holidays
    print("   🎉 Identifying Holidays...")
    unique_years = df_train['time'].dt.year.unique().tolist()
    try:
        de_holidays = holidays.Germany(subdiv='HE', years=unique_years)
        df_train['is_holiday'] = df_train['time'].dt.date.apply(lambda x: 1 if x in de_holidays else 0)
    except:
        df_train['is_holiday'] = 0

    # External Factors
    df_train = process_weather_data(df_train)
    df_train = process_construction_data(df_train)
    df_train = process_strike_data(df_train)

    df_train.to_csv(OUTPUT_FILE, index=False)
    print(f"   ✅ SUCCESS: Saved to {OUTPUT_FILE}")