import pickle
import pandas as pd
import os

class SmartCommuteAdvisor:
    def __init__(self, model_path="delay_model_v2.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found! Please run '05_train_ml_model.py' first.")
            
        print("⏳ Loading AI Brain...")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.station_map = data['map']
        
        print("✅ EBS Advisor Ready.")

    def get_station_id(self, name_input):
        # Flexible matching for inputs
        name_input = name_input.lower()
        if name_input in ['a', 'burg', 'campus a']: return 5 # Oestrich-Winkel
        if name_input in ['b', 'schloss', 'campus b']: return 6 # Hattenheim
        
        for name, id_val in self.station_map.items():
            if name_input in name.lower():
                return id_val
        return None

    def get_risk(self, source_id, dest_id, day, hour, train_type="RB"):
        # Logic: Determine Direction based on Source vs Dest
        if source_id < dest_id: direction = 1 # Westbound
        else: direction = 0 # Eastbound

        # Create input for the model
        input_data = pd.DataFrame({
            'weekday': [day],
            'hour': [hour],
            'train_type': [train_type],
            'station_id': [dest_id],
            'direction': [direction]
        })
        
        # Get probability (Index 1 is the probability of "Delay=1")
        return self.model.predict_proba(input_data)[0][1]

    def recommend_trip(self, source_name, campus_name, day, class_hour, class_min):
        source_id = self.get_station_id(source_name)
        dest_id = self.get_station_id(campus_name)
        
        if source_id is None: return f"❌ Unknown Source: '{source_name}'"
        if dest_id is None: return f"❌ Unknown Campus: '{campus_name}'"

        # OPTION A: The "Just in Time" Train (Arrives in the hour prior to class)
        # e.g. Class at 10:15 -> Check Hour 9
        opt1_hour = class_hour - 1 if class_hour > 0 else 23
        opt1_risk = self.get_risk(source_id, dest_id, day, opt1_hour)
        
        # OPTION B: The "Safe Buffer" Train (Arrives 2 hours prior)
        # e.g. Class at 10:15 -> Check Hour 8
        opt2_hour = class_hour - 2 if class_hour > 1 else 22
        opt2_risk = self.get_risk(source_id, dest_id, day, opt2_hour)

        train_name = "RB10" # Default

        # --- THE "SOUL STARE" LOGIC ---
        header = f"\n🔎 ANALYSIS for {day} (Class @ {class_hour}:{class_min:02d})"
        stats = (f"   Option A ({opt1_hour}:00 arrival): {opt1_risk:.0%} Risk\n"
                 f"   Option B ({opt2_hour}:00 arrival): {opt2_risk:.0%} Risk")
        
        if opt1_risk < 0.25:
            advice = (f"✅ GREEN LIGHT: Take the {train_name} arriving at {opt1_hour}:00.\n"
                      f"   The risk is low ({opt1_risk:.0%}). You'll be fine.")
        
        elif opt1_risk < 0.45:
            advice = (f"⚠️ CAUTION: The {opt1_hour}:00 train is risky ({opt1_risk:.0%}).\n"
                      f"   You should probably take the {opt2_hour}:00 one to be safe.")
        
        else:
            advice = (f"🛑 DANGER ZONE: Do NOT take the {opt1_hour}:00 train (Risk: {opt1_risk:.0%})!\n"
                      f"   👉 YOU MUST TAKE the {train_name} arriving at {opt2_hour}:00.\n\n"
                      f"   👻 WARNING: If you are late, the professor will stare into your soul.\n"
                      f"   Don't be that person.")

        return f"{header}\n{stats}\n\n{advice}"

# --- INTERACTIVE LOOP ---
if __name__ == "__main__":
    advisor = SmartCommuteAdvisor()
    
    print("\n🎓 --- EBS PERSONAL COMMUTE ADVISOR ---")
    print("Commands: Type 'exit' to quit.")
    
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