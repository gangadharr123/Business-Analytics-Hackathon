"""
05_train_ml_model.py
--------------------
BUSINESS ANALYTICS HACKATHON - MODEL TRAINING
Step 3: The "Battle of the Algorithms" (Compressed Save)

Description:
    Trains and compares 3 algorithms.
    Saves the winner using JOBLIB COMPRESSION to avoid GitHub file size limits.

Author: Gangadhar
"""

import pandas as pd
import joblib  # <--- CHANGED FROM PICKLE TO JOBLIB
import numpy as np
import os
import warnings

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
CSV_PATH = os.path.join(ROOT_DIR, 'data', 'ebs_commute_data_enriched.csv')
MODEL_PATH = os.path.join(SRC_DIR, 'delay_model_v2.pkl')

STATION_MAP = {
    'Frankfurt(Main)Hbf': 0, 'Frankfurt-Höchst': 1, 'Wiesbaden Hbf': 2,
    'Wiesbaden-Biebrich': 3, 'Eltville': 4, 'Oestrich-Winkel': 5,
    'Hattenheim': 6, 'Geisenheim': 7, 'Rüdesheim(Rhein)': 8
}

def get_direction(final_dest):
    return 0 if 'Frankfurt' in str(final_dest) else 1

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def get_feature_names(preprocessor, numeric_features):
    try:
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        return np.concatenate([cat_features, numeric_features])
    except:
        return [f"Feature {i}" for i in range(100)]

def print_top_features(model, feature_names, model_name, top_n=5):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return

        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(top_n)
        
        print(f"\n      🔎 TOP {top_n} FEATURES for {model_name}:")
        for i, row in feat_df.iterrows():
            name = row['Feature'].replace('weekday_', '').replace('station_id_', 'Station ').replace('train_type_', '')
            print(f"         {i+1}. {name:<25} ({row['Importance']:.3f})")
    except:
        pass

if __name__ == "__main__":
    print_separator("PIPELINE START: BATTLE OF THE ALGORITHMS")
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: Data not found at {CSV_PATH}. Run Step 2.")
        exit()

    df = pd.read_csv(CSV_PATH)
    df['is_delayed'] = (df['delay_in_min'] > 3).astype(int)
    df['station_id'] = df['station_name'].map(STATION_MAP)
    df = df.dropna(subset=['station_id'])
    df['direction'] = df['final_destination_station'].apply(get_direction)

    weather_cols = [
        'temp_c', 'humidity_pct', 'precip_mm', 'rain_mm', 'snow_cm',
        'feels_like_c', 'wind_gusts_kmh', 'wind_speed_kmh', 'wind_dir_10m'
    ]
    for c in weather_cols:
        if c not in df.columns: df[c] = 0
            
    base_features = [
        'weekday', 'hour', 'train_type', 'station_id', 'direction', 
        'is_holiday', 'construction_impact', 'strike_impact'
    ]
    features = base_features + [c for c in weather_cols if c in df.columns]

    X = df[features]
    y = df['is_delayed']
    
    print(f"   📊 Training on {X.shape[0]} trips with {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_cols = ['weekday', 'train_type', 'station_id']
    num_cols = [c for c in features if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])

    models = [
        {'name': 'Decision Tree', 'clf': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)},
        {'name': 'Logistic Regression', 'clf': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)},
        {'name': 'Random Forest', 'clf': RandomForestClassifier(n_estimators=200, max_depth=20, class_weight={0:1, 1:3}, random_state=42, n_jobs=-1)}
    ]

    results = []
    print_separator("ROUND 1: DEEP DIVE ANALYSIS")

    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor, num_cols)

    for m in models:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', m['clf'])])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"\n   🥊 MODEL: {m['name'].upper()}")
        print(f"      Accuracy: {acc:.1%}")
        print(f"      Recall:   {recall:.1%} (Safety Score)")
        print(f"      ROC-AUC:  {auc:.3f}")
        print(f"      Confusion Matrix: [ TN: {tn:<5} | FP: {fp:<5} ]  [ FN: {fn:<5} | TP: {tp:<5} ]")
        
        print_top_features(m['clf'], feature_names, m['name'])
        print("-" * 60)
        
        results.append({'name': m['name'], 'pipeline': pipeline, 'recall': recall, 'auc': auc})

    print_separator("ROUND 2: AND THE WINNER IS...")
    best_score = -1
    winner = None
    
    for r in results:
        score = (r['recall'] * 0.7) + (r['auc'] * 0.3)
        if score > best_score:
            best_score = score
            winner = r

    print(f"\n   🏆 CHAMPION: {winner['name'].upper()}")

    # --- SAVE WITH COMPRESSION ---
    joblib.dump({'model': winner['pipeline'], 'map': STATION_MAP}, MODEL_PATH, compress=3) # <--- COMPRESS=3
    print(f"\n   💾 Saved {winner['name']} to {MODEL_PATH} (Compressed)")