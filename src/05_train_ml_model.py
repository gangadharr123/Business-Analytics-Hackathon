import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 1. CONFIGURATION ---
STATION_MAP = {
    'Frankfurt(Main)Hbf': 0,
    'Frankfurt-Höchst': 1,
    'Wiesbaden Hbf': 2,
    'Wiesbaden-Biebrich': 3,
    'Eltville': 4,
    'Oestrich-Winkel': 5, # Campus A
    'Hattenheim': 6,      # Campus B
    'Geisenheim': 7,
    'Rüdesheim(Rhein)': 8
}

def get_direction(final_dest):
    if 'Frankfurt' in str(final_dest): return 0 
    return 1

# --- 2. PREPARE DATA ---
print("⏳ Loading data...")
# Ensure this points to the new CSV you just created
df = pd.read_csv("src/ebs_commute_data.csv")

# Create Target
df['is_delayed'] = (df['delay_in_min'] > 3).astype(int)

# Feature Engineering
df['station_id'] = df['station_name'].map(STATION_MAP)
df = df.dropna(subset=['station_id'])
df['direction'] = df['final_destination_station'].apply(get_direction)

# Features
features = ['weekday', 'hour', 'train_type', 'station_id', 'direction']
X = df[features]
y = df['is_delayed']

# --- 3. BUILD PIPELINE (With The Fix) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['weekday', 'train_type']),
        ('num', 'passthrough', ['hour', 'station_id', 'direction'])
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # FIX: class_weight='balanced' forces the model to capture more delays
    ('model', RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42,
        class_weight='balanced'  # <--- THIS IS THE KEY CHANGE
    ))
])

# --- 4. TRAIN ---
print(f"🏋️ Training with Class Balancing on {len(X)} trips...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# --- 5. EVALUATE ---
y_pred = pipeline.predict(X_test)
print(f"\n✅ Model Trained!")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# New: Print Confusion Matrix to see exactly how many delays we caught
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nCaught Delays (True Positives): {tp}")
print(f"Missed Delays (False Negatives): {fn}")
print(f"Recall Score: {tp / (tp + fn):.2%}")

# --- 6. SAVE ---
package = {'model': pipeline, 'map': STATION_MAP}
with open('delay_model_v2.pkl', 'wb') as f:
    pickle.dump(package, f)
print("💾 Model saved.")