"""
06_smart_commute_tool.py
------------------------
BUSINESS ANALYTICS HACKATHON - DEPLOYMENT
Step 4: Interactive Commuter Advisor

Description:
    The user-facing tool. Loads the trained ML model and allows
    students to check risk probabilities.
    
    Key Feature: "The Soul Stare Logic"
    - Calculates risk for the 'Latest Possible Train' (Option A).
    - If risk > 45%, recommends 'Earlier Buffer Train' (Option B).

Author: Gangadhar
"""

import pickle
import pandas as pd
import os

class SmartCommuteAdvisor:
    def __init__(self, model_filename="delay_model_v2.pkl"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, model_filename)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model not found at: {self.model_path}")
            
        print(f"⏳ Loading AI Brain from: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.station_map = data['map']
        
        print("✅ EBS Advisor Ready.")

    def get_station_id(self, name_input):
        name_input = name_input.lower()
        if name_input in ['a', 'burg', 'campus a']: return 5
        if name_input in ['b', 'schloss', 'campus b']: return 6
        for name, id_val in self.station_map.items():
            if name_input in name.lower():
                return id_val
        return None

    def get_risk(self, source_id, dest_id, day, hour, train_type="RB"):
        # 1. Determine Direction
        if source_id < dest_id: direction = 1 # Westbound (to Rheingau)
        else: direction = 0 # Eastbound (to Frankfurt)

        # 2. Construct Input Vector
        # CRITICAL: We must provide ALL features the model was trained on.
        # For this demo, we assume "Standard Conditions" (No holiday/strike)
        # unless we explicitly build a date-picker.
        input_data = pd.DataFrame({
            'weekday': [day],
            'hour': [hour],
            'train_type': [train_type],
            'station_id': [dest_id],
            'direction': [direction],
            'is_holiday': [0],          # Default: Not a holiday
            'construction_impact': [0], # Default: No construction
            'strike_impact': [0]        # Default: No strike
        })
        
        # 3. Predict Probability of Delay
        # [0] is "No Delay", [1] is "Delay"
        return self.model.predict_proba(input_data)[0][1]

    def recommend_trip(self, source_name, campus_name, day, class_hour, class_min):
        source_id = self.get_station_id(source_name)
        dest_id = self.get_station_id(campus_name)
        
        if source_id is None: return f"❌ Unknown Source: '{source_name}'"
        if dest_id is None: return f"❌ Unknown Campus: '{campus_name}'"

        # Logic: Check train arriving 1 hour before class (Option A) vs 2 hours (Option B)
        # Assuming class starts e.g. 10:15
        opt1_hour = class_hour - 1 if class_hour > 0 else 23
        opt2_hour = class_hour - 2 if class_hour > 1 else 22
        
        opt1_risk = self.get_risk(source_id, dest_id, day, opt1_hour)
        opt2_risk = self.get_risk(source_id, dest_id, day, opt2_hour)

        train_name = "RB10" # Default student train

        # --- THE ADVISOR LOGIC ---
        header = f"\n🔎 ANALYSIS for {day} (Class @ {class_hour}:{class_min:02d})"
        stats = (f"   Option A ({opt1_hour}:00 arrival): {opt1_risk:.0%} Risk\n"
                 f"   Option B ({opt2_hour}:00 arrival): {opt2_risk:.0%} Risk")
        
        if opt1_risk < 0.25:
            advice = (f"✅ GREEN LIGHT: Take the {train_name} arriving at {opt1_hour}:00.\n"
                      f"   Risk is low. Enjoy your coffee.")
        
        elif opt1_risk < 0.45:
            advice = (f"⚠️ CAUTION: The {opt1_hour}:00 train is risky ({opt1_risk:.0%}).\n"
                      f"   Consider the earlier train if you have an exam.")
        
        else:
            advice = (f"🛑 DANGER ZONE: Do NOT take the {opt1_hour}:00 train (Risk: {opt1_risk:.0%})!\n"
                      f"   👉 YOU MUST TAKE the {train_name} arriving at {opt2_hour}:00.\n\n"
                      f"   👻 WARNING: If you are late, the professor will stare into your soul.\n"
                      f"   Don't risk it.")

        return f"{header}\n{stats}\n\n{advice}"

if __name__ == "__main__":
    advisor = SmartCommuteAdvisor()
    
    print("\n🎓 --- EBS PERSONAL COMMUTE ADVISOR ---")
    print("Type 'exit' to quit.")
    
    while True:
        print("\n-------------------------------------------")
        try:
            s = input("Where do you live? (e.g. Wiesbaden): ").strip()
            if s.lower() in ['exit', 'quit']: break
            
            c = input("Which Campus? (Burg / Schloss): ").strip()
            day = input("Class Day (e.g. Monday): ").strip().capitalize()
            
            print("When does class start?")
            h = int(input("   Hour (0-23): "))
            m = int(input("   Minute (0-59): "))
            
            print(advisor.recommend_trip(s, c, day, h, m))
                
        except ValueError:
            print("❌ Error: Please enter numbers for the time.")
        except Exception as e:
            print(f"❌ Error: {e}")