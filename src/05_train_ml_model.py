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

# --- PATH SETUP ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
# Update: Look for the enriched file in the DATA folder
CSV_PATH = os.path.join(ROOT_DIR, 'data', 'ebs_commute_data_enriched.csv')
MODEL_PATH = os.path.join(SRC_DIR, 'delay_model_v2.pkl')

# --- CONFIGURATION ---
STATION_MAP = {
    'Frankfurt(Main)Hbf': 0,
    'Frankfurt-Höchst': 1,
    'Wiesbaden Hbf': 2,
    'Wiesbaden-Biebrich': 3,
    'Eltville': 4,
    'Oestrich-Winkel': 5,
    'Hattenheim': 6,
    'Geisenheim': 7,
    'Rüdesheim(Rhein)': 8
}

def get_direction(final_dest):
    if 'Frankfurt' in str(final_dest): return 0 
    return 1

# --- LOAD DATA ---
if not os.path.exists(CSV_PATH):
    print(f"❌ Error: {CSV_PATH} not found.\n   Please run '02_enrich_data.py' to generate it in the data folder.")
    exit()

print(f"⏳ Loading enriched data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Target: Delay > 3 minutes
df['is_delayed'] = (df['delay_in_min'] > 3).astype(int)

# Feature Engineering
df['station_id'] = df['station_name'].map(STATION_MAP)
df = df.dropna(subset=['station_id'])
df['direction'] = df['final_destination_station'].apply(get_direction)

# --- FEATURES (No Weather, Yes Events) ---
features = [
    'weekday', 'hour', 'train_type', 'station_id', 'direction', 
    'is_holiday', 'construction_impact', 'strike_impact'
]

X = df[features]
y = df['is_delayed']

# --- BUILD PIPELINE ---
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
    ('model', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced'))
])

# --- TRAIN ---
print(f"🏋️ Training model on {len(X)} trips...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = pipeline.predict(X_test)
print(f"\n✅ Model Trained!")
print(classification_report(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Caught Delays (Recall): {tp / (tp + fn):.2%}")

# --- SAVE ---
package = {'model': pipeline, 'map': STATION_MAP}
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(package, f)
print(f"💾 Model saved to {MODEL_PATH}")