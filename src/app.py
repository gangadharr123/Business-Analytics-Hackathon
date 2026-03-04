"""Streamlit Web Application for EBS Smart Commute"""

import streamlit as st
import datetime
import importlib.util
import sys
from pathlib import Path

# Add local src/ path so Streamlit can import project modules when launched from repo root.
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from config import TARGET_STATIONS
import importlib
smart_tool = importlib.import_module("06_smart_commute_tool")
SmartCommuteAdvisor = smart_tool.SmartCommuteAdvisor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EBS Commute Advisor", page_icon="🚆", layout="centered")

# --- MODEL LOADING ---
@st.cache_resource
def load_advisor():
    return SmartCommuteAdvisor()

try:
    advisor = load_advisor()
except Exception as e:
    st.error(f"Failed to load the AI model. Ensure you have run '05_train_ml_model.py' first. Error: {e}")
    st.stop()

# --- UI HEADER ---
st.title("🎓 EBS Smart Commute Advisor")
st.markdown("Never face the Professor's *Soul Stare*. Enter your class details below to get an AI-powered travel recommendation.")

# --- USER INPUTS ---
with st.form("commute_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        source = st.selectbox("Starting Station", sorted(TARGET_STATIONS), index=sorted(TARGET_STATIONS).index("Wiesbaden Hbf"))
        day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    
    with col2:
        dest = st.selectbox("Campus Destination", ["Oestrich-Winkel (Campus A / Burg)", "Hattenheim (Campus B / Schloss)"])
        class_time = st.time_input("Class Start Time", value=datetime.time(9, 0))
    
    st.markdown("---")
    exam_mode = st.checkbox("🚨 I have a critical Exam / Presentation today (Low Risk Tolerance)", value=False)
    
    # Optional context flag for large public events that can increase dwell times.
    event_mode = st.checkbox("🏟️ Major Event Today (e.g., Match, Festival, Trade Fair)", value=False)
    
    submitted = st.form_submit_button("Find My Train 🔍")

# --- PREDICTION LOGIC & OUTPUT ---
if submitted:
    class_hour = class_time.hour
    
    # Convert checkbox state to the binary feature expected by the model.
    event_flag = 1 if event_mode else 0
    
    # Evaluate two candidate departures that arrive roughly 1h and 2h before class.
    opt1_hour = class_hour - 1 if class_hour >= 1 else 23
    opt2_hour = class_hour - 2 if class_hour >= 2 else 22
    
    # Score both options with identical contextual inputs.
    prob1, label1, buf1, feat1 = advisor.get_risk(source, dest, day, opt1_hour, has_event=event_flag)
    prob2, label2, buf2, feat2 = advisor.get_risk(source, dest, day, opt2_hour, has_event=event_flag)
    
    # Lower tolerance when users mark exam/presentation mode.
    safe_threshold = 0.25 if exam_mode else 0.40
    
    st.header("📋 AI Recommendation")
    
    if prob1 <= safe_threshold:
        st.success(f"✅ **Take the {opt1_hour}:00 Train.**")
        st.write(f"The risk of a catastrophic delay is low ({prob1:.1%}). You'll arrive with plenty of time to grab a coffee with milk to start your day before heading to class.")
        target_feat = feat1
    else:
        st.error(f"🛑 **AVOID the {opt1_hour}:00 Train!**")
        st.write(f"This train has a high risk of delay ({prob1:.1%}). To guarantee you make it to class on time, you should take the earlier train arriving at **{opt2_hour}:00** (Risk: {prob2:.1%}).")
        target_feat = feat1
        
    # --- EXPLAINABILITY SECTION ---
    with st.expander("🤖 Why did the AI make this prediction?"):
        st.write("Our model analyzes historical Deutsche Bahn delays. For your specific route and time, it flagged the following risk factors:")
        
        factors_found = False
        if target_feat.get("is_rush_hour") == 1:
            st.warning("🚉 **Platform Congestion:** You are traveling during peak commuter hours.")
            factors_found = True
        if target_feat.get("is_freezing") == 1:
            st.info("❄️ **Freezing Conditions:** Temperatures at or below 0°C increase the risk of frozen rail switches.")
            factors_found = True
        if target_feat.get("has_precipitation") == 1:
            st.info("🌧️ **Precipitation:** Rain or snow is causing slower boarding times and potential track issues.")
            factors_found = True
        if target_feat.get("high_winds") == 1:
            st.warning("💨 **High Winds:** Deutsche Bahn enforces speed restrictions during heavy gusts.")
            factors_found = True
            
        # Explain event-based risk when the feature is active.
        if target_feat.get("has_event") == 1:
            st.warning("🎟️ **Crowd Surge:** Major events dramatically increase boarding dwell times, causing localized micro-delays.")
            factors_found = True
            
        if not factors_found:
            st.success("✨ Standard travel conditions detected. No extreme external risk factors flagged.")