import pandas as pd
import pyarrow.parquet as pq
import os

# 1. SETUP: Define paths
# Input: The big Parquet file
file_path = r"C:\Users\Gangadhar\OneDrive\Study\EBS\Hackathon\Business-Analytics-Hackathon\data\data-2025-12.parquet"

# Output: The processed CSV (Saving directly to 'src' folder to match your training script)
output_path = r"C:\Users\Gangadhar\OneDrive\Study\EBS\Hackathon\Business-Analytics-Hackathon\src\ebs_commute_data.csv"

# Stations relevant to EBS
target_stations = [
    "Frankfurt(Main)Hbf",
    "Frankfurt-Höchst",
    "Wiesbaden Hbf",
    "Wiesbaden-Biebrich",
    "Eltville",
    "Oestrich-Winkel", # Campus A
    "Hattenheim",      # Campus B
    "Geisenheim",
    "Rüdesheim(Rhein)",
    "Mainz Hbf"
]

print("⏳ Re-generating EBS dataset (including Destination info)...")

try:
    # 2. LOAD - Explicitly including 'final_destination_station'
    table = pq.read_table(
        file_path, 
        columns=[
            'station_name', 
            'time', 
            'delay_in_min', 
            'train_type', 
            'final_destination_station'  # <--- CRITICAL COLUMN
        ]
    )
    df = table.to_pandas()
    
    # 3. FILTER
    df_filtered = df[df['station_name'].isin(target_stations)].copy()
    
    # 4. CLEAN
    df_filtered['time'] = pd.to_datetime(df_filtered['time'])
    df_filtered['hour'] = df_filtered['time'].dt.hour
    df_filtered['weekday'] = df_filtered['time'].dt.day_name()
    
    # 5. SAVE
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_filtered.to_csv(output_path, index=False)
    
    print(f"✅ Success! Saved {len(df_filtered):,} rows to:")
    print(f"   '{output_path}'")
    print("   Columns:", list(df_filtered.columns))

except Exception as e:
    print(f"❌ Error: {e}")