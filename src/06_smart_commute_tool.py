"""
06_smart_commute_tool.py
------------------------
BUSINESS ANALYTICS HACKATHON - DEPLOYMENT
Step 4: Interactive Commuter Advisor

Description:
    The user-facing tool. Loads the trained ML model and allows
    students to check risk probabilities.
    
    NOW USES JOBLIB TO LOAD COMPRESSED MODELS.

Author: Gangadhar
"""

import joblib  # <--- CHANGED FROM PICKLE TO JOBLIB
import pandas as pd
import os
import datetime

class SmartCommuteAdvisor:
    def __init__(self, model_filename="delay_model_v2.pkl"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, model_filename)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model not found at: {self.model_path}. Run Step 3 first.")
            
        print(f"⏳ Loading AI Brain from: {self.model_path}...")
        try:
            # Load compressed model using joblib
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.station_map = data['map']
            print("✅ EBS Advisor Ready.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e

    def get_risk(self, source, dest, day, hour, train_type="RB"):
        # Helper to get station IDs (simplified for demo)
        def get_id(name):
            name = name.lower()
            if 'frankfurt' in name: return 0
            if 'wiesbaden' in name: return 2
            if 'eltville' in name: return 4
            if 'oestrich' in name or 'burg' in name: return 5 # Campus A
            if 'hattenheim' in name or 'schloss' in name: return 6 # Campus B
            if 'geisenheim' in name: return 7
            if 'rudesheim' in name: return 8
            return 2 # Default to Wiesbaden
            
        s_id = get_id(source)
        d_id = get_id(dest)
        direction = 0 if s_id > d_id else 1 # Simple heuristic

        # Create input DataFrame
        input_data = pd.DataFrame({
            'weekday': [day],
            'hour': [hour],
            'train_type': [train_type],
            'station_id': [s_id],
            'direction': [direction],
            'is_holiday': [0], # Default
            'construction_impact': [0], # Default
            'strike_impact': [0], # Default
            'temp_c': [5.0], # Default winter temp
            'humidity_pct': [80],
            'precip_mm': [0],
            'rain_mm': [0],
            'snow_cm': [0],
            'feels_like_c': [3.0],
            'wind_gusts_kmh': [15.0],
            'wind_speed_kmh': [10.0],
            'wind_dir_10m': [200]
        })

        # Predict Probability
        prob = self.model.predict_proba(input_data)[0][1]
        return prob

if __name__ == "__main__":
    advisor = SmartCommuteAdvisor()
    
    print("\n--- TEST: Friday 6 PM to Campus ---")
    risk = advisor.get_risk("Frankfurt", "Burg", "Friday", 18)
    print(f"Risk: {risk:.1%}")
    
    if risk > 0.45:
        print("⚠️  WARNING: High Delay Risk! Take an earlier train.")
    else:
        print("✅ Safe to travel.")