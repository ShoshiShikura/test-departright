import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load Model
with open('sawit_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🚛 SawitClear: Pinch-Point Forecaster")
st.write("**Target Bottleneck:** Telupid Mountain Pass - KM 42")

# User Input Panel
col1, col2, col3 = st.columns(3)
with col1:
    planned_hour = st.slider("Planned Departure Hour", 6, 19, 8)
with col2:
    day_of_week = st.selectbox("Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
with col3:
    weather = st.selectbox("Current Weather", [0, 1, 2], format_func=lambda x: ["Clear", "Light Rain", "Heavy Rain"][x])

# AI Inference & Temporal Optimisation
times_to_check = [planned_hour - 1, planned_hour, planned_hour + 1]
predictions = []

for h in times_to_check:
    if 6 <= h <= 19:
        prob = model.predict_proba(pd.DataFrame([[h, day_of_week, weather]], columns=['hour_of_day', 'day_of_week', 'rain_intensity']))[0][1]
        predictions.append((h, prob))

# Visualise the Results
st.subheader("Temporal Routing Recommendation")

current_risk = next(p for h, p in predictions if h == planned_hour)

if current_risk > 0.6:
    st.error(f"High Convoy Risk Detected ({int(current_risk*100)}%) at {planned_hour}:00")
else:
    st.success(f"Path Clear: Low Risk ({int(current_risk*100)}%) at {planned_hour}:00")

# Smart Recommendation Logic
best_time = min(predictions, key=lambda x: x[1])
if best_time[0] != planned_hour and best_time[1] < current_risk:
    st.info(f"💡 **AI Suggestion:** Adjust departure to **{best_time[0]}:00** to reduce encounter probability to {int(best_time[1]*100)}%.")

# Crowdsource Feature (PoC)
if st.button("🚨 I am stuck behind a lorry here right now"):
    st.warning("Live incident reported! Recalibrating risk scores for the next 45 minutes...")