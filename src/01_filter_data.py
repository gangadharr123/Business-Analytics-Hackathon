"""
01_filter_data.py
-----------------
BUSINESS ANALYTICS HACKATHON - DATA ENGINEERING PIPELINE
Step 1: Raw Data Filtering

Description:
    Ingests raw parquet train data and filters for the specific 
    Frankfurt-Rheingau commuter corridor relevant to EBS University.
    Reduces data volume from ~15M rows to ~50k relevant rows.

Author: Gangadhar
Date: 2026-02-13
"""

import pandas as pd
import pyarrow.parquet as pq
import os

# --- 1. CONFIGURATION ---
# Target Stations: The "EBS Commuter Line" (Frankfurt -> Rheingau)
TARGET_STATIONS = [
    "Frankfurt(Main)Hbf", "Frankfurt-Höchst", "Wiesbaden Hbf", 
    "Wiesbaden-Biebrich", "Eltville", "Oestrich-Winkel", 
    "Hattenheim", "Geisenheim", "Rüdesheim(Rhein)", "Mainz Hbf"
]

# --- 2. PATH SETUP ---
# Dynamic paths to ensure code runs on any machine
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Input/Output
RAW_DATA_FILE = os.path.join(DATA_DIR, 'data-2025-12.parquet')
OUTPUT_FILE = os.path.join(SRC_DIR, 'ebs_commute_data.csv') # Intermediate file stays in src for processing

def filter_commuter_data():
    print(f"🔄 PIPELINE START: Filtering raw data...")
    print(f"   📂 Reading: {RAW_DATA_FILE}")

    if not os.path.exists(RAW_DATA_FILE):
        print(f"   ❌ ERROR: Raw data not found at {RAW_DATA_FILE}")
        return

    try:
        # Load specific columns to optimize memory usage
        # 'final_destination_station' is CRITICAL for determining direction
        table = pq.read_table(
            RAW_DATA_FILE, 
            columns=[
                'station_name', 'time', 'delay_in_min', 
                'train_type', 'final_destination_station'
            ]
        )
        df = table.to_pandas()
        
        # Filter for relevant stations
        df_filtered = df[df['station_name'].isin(TARGET_STATIONS)].copy()
        
        # Feature Engineering: Extract time components
        df_filtered['time'] = pd.to_datetime(df_filtered['time'])
        df_filtered['hour'] = df_filtered['time'].dt.hour
        df_filtered['weekday'] = df_filtered['time'].dt.day_name()
        
        # Save processed dataset
        df_filtered.to_csv(OUTPUT_FILE, index=False)
        
        print(f"   ✅ SUCCESS: Filtered {len(df_filtered):,} rows.")
        print(f"   💾 Saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"   ❌ FATAL ERROR: {e}")

if __name__ == "__main__":
    filter_commuter_data()