"""
05_train_ml_model.py
--------------------
BUSINESS ANALYTICS HACKATHON - MODEL TRAINING
Step 3: Machine Learning Pipeline

Description:
    Trains a Random Forest Classifier to predict delay probability (>3 mins).
    Uses 'Balanced Class Weights' to solve the imbalance problem (fewer delays than on-time trains).
    
    Features Used:
    - Temporal: Weekday, Hour
    - Spatial: Station, Direction
    - External: Construction Impact, Strikes, Holidays

Author: Gangadhar
"""

import pandas as pd
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 1. CONFIGURATION ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
CSV_PATH = os.path.join(ROOT_DIR, 'data', 'ebs_commute_data_enriched.csv')
MODEL_PATH = os.path.join(SRC_DIR, 'delay_model_v2.pkl')

STATION_MAP = {
    'Frankfurt(Main)Hbf': 0, 'Frankfurt-Höchst': 1, 'Wiesbaden Hbf': 2,
    'Wiesbaden-Biebrich': 3, 'Eltville': 4, 'Oestrich-Winkel': 5, # Campus A
    'Hattenheim': 6, # Campus B
    'Geisenheim': 7, 'Rüdesheim(Rhein)': 8
}

def get_direction(final_dest):
    """Heuristic: Determines if train is heading East (Frankfurt) or West (Rheingau)"""
    if 'Frankfurt' in str(final_dest): return 0 
    return 1

if __name__ == "__main__":
    print("🔄 PIPELINE START: Model Training...")
    
    if not os.path.exists(CSV_PATH):
        print(f"   ❌ ERROR: Data not found at {CSV_PATH}. Run Step 2.")
        exit()

    # Load Data
    df = pd.read_csv(CSV_PATH)
    
    # Target Definition: Delay > 3 minutes
    df['is_delayed'] = (df['delay_in_min'] > 3).astype(int)

    # Feature Engineering
    df['station_id'] = df['station_name'].map(STATION_MAP)
    df = df.dropna(subset=['station_id'])
    df['direction'] = df['final_destination_station'].apply(get_direction)

    # --- FEATURE SELECTION ---
    # Must match the columns generated in Step 2
    features = [
        'weekday', 'hour', 'train_type', 'station_id', 'direction', 
        'is_holiday', 'construction_impact', 'strike_impact'
    ]
    X = df[features]
    y = df['is_delayed']

    # --- PIPELINE BUILD ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['weekday', 'train_type']),
            ('num', 'passthrough', [
                'hour', 'station_id', 'direction', 
                'is_holiday', 'construction_impact', 'strike_impact'
            ])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # class_weight='balanced' is CRITICAL for catching delays
        ('model', RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, class_weight='balanced'
        ))
    ])

    # --- TRAIN & EVALUATE ---
    print(f"   🏋️ Training on {len(X)} trips...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print("\n   📊 MODEL RESULTS:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.1%}")
    print(f"   Recall (Caught Delays): {tp / (tp + fn):.1%}")
    
    # Save
    package = {'model': pipeline, 'map': STATION_MAP}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(package, f)
    print(f"   💾 Model saved to {MODEL_PATH}")