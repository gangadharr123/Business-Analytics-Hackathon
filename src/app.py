"""Streamlit Web Application for EBS Smart Commute"""

import streamlit as st
import datetime
import importlib.util
import sys
from pathlib import Path

# Add local src/ path so Streamlit can import project modules when launched from repo root.
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from config import SAFE_THRESHOLD_EXAM, SAFE_THRESHOLD_NORMAL, STATION_MAP
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
        _stations = sorted(STATION_MAP.keys())
        source = st.selectbox("Starting Station", _stations, index=_stations.index("Wiesbaden Hbf"))
        day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

    with col2:
        dest = st.selectbox("Campus Destination", ["Oestrich-Winkel", "Hattenheim"])
        class_time = st.time_input("Class Start Time", value=datetime.time(9, 0))
    
    st.markdown("---")
    exam_mode = st.checkbox("🚨 I have a critical Exam / Presentation today (Low Risk Tolerance)", value=False)
    
    # Optional context flag for large public events that can increase dwell times.
    event_mode = st.checkbox("🏟️ Major Event Today (e.g., Match, Festival, Trade Fair)", value=False)
    
    submitted = st.form_submit_button("Find My Train 🔍")

# --- PREDICTION LOGIC & OUTPUT ---
if submitted:
    class_hour = class_time.hour

    event_flag = 1 if event_mode else 0

    # Two candidate departures: 1h and 2h before class.
    opt1_hour = class_hour - 1 if class_hour >= 1 else 23
    opt2_hour = class_hour - 2 if class_hour >= 2 else 22

    prob1, label1, buf1, feat1 = advisor.get_risk(source, dest, day, opt1_hour, has_event=event_flag)
    prob2, label2, buf2, feat2 = advisor.get_risk(source, dest, day, opt2_hour, has_event=event_flag)

    safe_threshold = SAFE_THRESHOLD_EXAM if exam_mode else SAFE_THRESHOLD_NORMAL

    def _risk_label(p: float) -> tuple[str, str]:
        if p >= 0.60: return "HIGH",        "🔴"
        if p >= 0.40: return "MEDIUM",      "🟡"
        if p >= 0.25: return "LOW-MEDIUM",  "🟠"
        return             "LOW",           "🟢"

    level1, icon1 = _risk_label(prob1)
    level2, icon2 = _risk_label(prob2)

    st.header("📋 AI Recommendation")
    st.markdown("---")

    # Primary recommendation banner
    if prob1 <= safe_threshold:
        st.success(f"✅ **Take the {opt1_hour}:00 train.** Delay risk is within your tolerance ({prob1:.1%}).")
    else:
        st.error(f"🛑 **Avoid the {opt1_hour}:00 train** — delay risk ({prob1:.1%}) exceeds your tolerance.")
        if prob2 <= safe_threshold:
            st.info(f"➡️ Take the earlier **{opt2_hour}:00 train** instead — risk is only {prob2:.1%}.")
        else:
            st.warning(f"⚠️ Both options carry elevated risk. The {opt2_hour}:00 train ({prob2:.1%}) is the safer choice.")

    # Key metrics row
    st.markdown("#### Key Risk Indicators — {opt1_hour}:00 Train".replace("{opt1_hour}", str(opt1_hour)))
    m1, m2, m3 = st.columns(3)
    m1.metric("Delay Probability",    f"{prob1:.1%}")
    m2.metric("Risk Level",           f"{icon1} {level1}")
    m3.metric("Recommended Buffer",   f"{buf1} min",
              help="Arrive this many minutes before your class to absorb a typical delay.")

    # Visual probability gauge
    st.caption(f"Risk gauge — delay probability for the {opt1_hour}:00 departure:")
    st.progress(min(prob1, 1.0))

    # Side-by-side option comparison
    st.markdown("---")
    st.subheader("🕐 Option Comparison")
    c1, c2 = st.columns(2)

    with c1:
        rec1 = "Recommended" if prob1 <= safe_threshold else "Not Recommended"
        st.markdown(f"**{opt1_hour}:00 Train** — *{rec1}*")
        st.metric("Delay Risk",     f"{prob1:.1%}")
        st.metric("Buffer Needed",  f"{buf1} min")
        st.caption(f"Risk level: {icon1} {level1}")

    with c2:
        rec2 = "Recommended" if prob2 <= safe_threshold else "Not Recommended"
        delta_pct = f"{prob2 - prob1:+.1%}"
        st.markdown(f"**{opt2_hour}:00 Train** — *{rec2}*")
        st.metric("Delay Risk",     f"{prob2:.1%}", delta=delta_pct, delta_color="inverse")
        st.metric("Buffer Needed",  f"{buf2} min")
        st.caption(f"Risk level: {icon2} {level2}")

    target_feat = feat1

    # Explainability
    st.markdown("---")
    with st.expander("🤖 Why did the AI make this prediction?"):
        st.write("The model was trained on historical Deutsche Bahn data for the Frankfurt–Rheingau corridor. "
                 "For your route and departure time, it detected the following risk factors:")

        factors_found = False
        if target_feat.get("is_rush_hour") == 1:
            st.warning("🚉 **Peak Hours:** You are traveling during rush hour — platform congestion increases dwell times.")
            factors_found = True
        if target_feat.get("is_freezing") == 1:
            st.info("❄️ **Freezing Conditions:** Temperatures ≤ 0°C risk frozen rail switches and slower operations.")
            factors_found = True
        if target_feat.get("has_precipitation") == 1:
            st.info("🌧️ **Precipitation:** Rain or snow slows boarding and can cause track issues.")
            factors_found = True
        if target_feat.get("high_winds") == 1:
            st.warning("💨 **High Winds:** DB enforces reduced speeds during gusts above 40 km/h.")
            factors_found = True
        if target_feat.get("has_event") == 1:
            st.warning("🎟️ **Major Event:** Crowd surges dramatically increase boarding dwell times.")
            factors_found = True

        if not factors_found:
            st.success("✨ Standard conditions — no elevated external risk factors detected for this departure.")