import pandas as pd
import os

class SimpleCommuteCalculator:
    def __init__(self, data_path=r"C:\Users\Gangadhar\OneDrive\Study\EBS\Hackathon\Business-Analytics-Hackathon\src\ebs_commute_data.csv"):
        if not os.path.exists(data_path):
            print(f"❌ Error: '{data_path}' not found. Please run your filtering script first.")
            return

        print("⏳ Loading Baseline Statistics...")
        self.df = pd.read_csv(data_path)
        
        # Define Delay Threshold (> 3 minutes)
        self.df['is_delayed'] = (self.df['delay_in_min'] > 3).astype(int)
        
        # 1. Pre-calculate the "Day x Hour" Risk Matrix
        self.risk_matrix = self.df.pivot_table(
            index='weekday', 
            columns='hour', 
            values='is_delayed', 
            aggfunc='mean'
        )
        print("✅ Baseline System Ready.")

    def get_risk(self, day, hour):
        try:
            # Simple Look-up: What is the average delay for this time slot?
            risk = self.risk_matrix.loc[day, hour]
            return risk
        except KeyError:
            return None

if __name__ == "__main__":
    tool = SimpleCommuteCalculator()
    
    print("\n📊 --- BASELINE CALCULATOR (No AI) ---")
    
    while True:
        try:
            print("\nCheck a time (or type 'exit'):")
            day = input("Day (e.g. Monday): ").strip().capitalize()
            if day.lower() in ['exit', 'quit']: break
            
            hour = int(input("Hour (0-23): "))
            
            risk = tool.get_risk(day, hour)
            
            if risk is not None:
                print(f"📈 Historical Delay Probability: {risk:.1%}")
            else:
                print("⚠️ No data for this specific time.")
                
        except ValueError:
            print("❌ Invalid input.")