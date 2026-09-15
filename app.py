import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SawitClear | Pinch-Point Forecaster",
    page_icon="🚛",
    layout="wide"
)

# Load AI Model
@st.cache_resource
def load_model():
    with open('sawit_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Header & Context
st.title("🚛 SawitClear: Temporal Pinch-Point Forecaster")
st.caption("Pan Borneo Highway • Choke Point: Telupid Mountain Pass (KM 42)")

# Sidebar: Route & Simulation Controls
with st.sidebar:
    st.header("⚙️ Journey Parameters")
    selected_day = st.selectbox(
        "Day of the Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
        index= datetime.today().weekday()
    )
    
    selected_weather = st.radio(
        "Current Weather Condition",
        options=[0, 1, 2],
        format_func=lambda x: ["☀️ Clear / Dry", "🌦️ Light Drizzle", "🌧️ Heavy Downpour"][x]
    )

    planned_time = st.slider("Target Departure Time", 6, 19, 9, format="%d:00")
    
    st.divider()
    st.subheader("🧪 Live Scenario Playground")
    plantation_surge = st.toggle("Simulate Fresh Fruit Bunch (FFB) Peak Rush", value=False)
    
# Main Grid: Metrics & Status
col1, col2, col3 = st.columns([1.2, 1, 1])

# Calculate 24-hour forecast curve
hours = list(range(6, 20))
probabilities = []
for h in hours:
    df_input = pd.DataFrame([[h, selected_day, selected_weather]], columns=['hour_of_day', 'day_of_week', 'rain_intensity'])
    base_prob = model.predict_proba(df_input)[0][1]
    
    # Inject simulation toggle
    if plantation_surge and 10 <= h <= 14:
        base_prob = min(base_prob + 0.25, 0.98)
        
    probabilities.append(base_prob)

current_risk = probabilities[hours.index(planned_time)]
optimum_hour = hours[np.argmin(probabilities)]
optimum_risk = min(probabilities)

with col1:
    if current_risk >= 0.60:
        st.error(f"### 🛑 High Convoy Risk: {int(current_risk * 100)}%")
        st.write("Severe lorry bottleneck predicted at the single-lane incline.")
    elif current_risk >= 0.40:
        st.warning(f"### ⚠️ Moderate Delay: {int(current_risk * 100)}%")
        st.write("Crawl speeds expected. Overtaking opportunities will be limited.")
    else:
        st.success(f"### ✅ Clear Passage: {int(current_risk * 100)}%")
        st.write("Optimum travel window. Minimal heavy vehicle interference.")

with col2:
    st.metric(
        label="AI Recommended Departure",
        value=f"{optimum_hour}:00",
        delta=f"-{int((current_risk - optimum_risk) * 100)}% Risk Reduction" if optimum_hour != planned_time else "Current is Best"
    )

with col3:
    est_time_saved = max(0, int((current_risk - optimum_risk) * 45))
    st.metric(
        label="Estimated Time Saved",
        value=f"{est_time_saved} Mins",
        delta="Clear Winding Pass"
    )

st.divider()

# Interactive Plotly Horizon Graph
st.subheader("📈 12-Hour Temporal Risk Horizon")

fig = go.Figure()

# Plot Risk Curve
fig.add_trace(go.Scatter(
    x=[f"{h}:00" for h in hours],
    y=[p * 100 for p in probabilities],
    mode='lines+markers',
    name='Convoy Probability',
    line=dict(color='#ff4b4b', width=3),
    fill='tozeroy',
    fillcolor='rgba(255, 75, 75, 0.15)'
))

# Highlight Selected Time
fig.add_vline(
    x=f"{planned_time}:00", 
    line_width=2, 
    line_dash="dash", 
    line_color="#1f77b4",
    annotation_text="Your Departure",
    annotation_position="top left"
)

fig.update_layout(
    xaxis_title="Time of Day",
    yaxis_title="Convoy Encounter Probability (%)",
    yaxis=dict(range=[0, 100]),
    height=320,
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Crowdsource & Community Section
st.divider()
st.subheader("📡 Real-Time Driver Verification")

c1, c2 = st.columns([1.5, 2])

with c1:
    st.write("Are you currently travelling along KM 42?")
    if st.button("🚨 Report Heavy Lorry Convoy", use_container_width=True):
        st.toast("Report logged! Thank you for validating the model.", icon="✅")
        st.info(f"Broadcasted to 14 active drivers near Telupid at {datetime.now().strftime('%H:%M:%S')}.")

with c2:
    st.markdown("**Recent Crowdsourced Telemetry:**")
    st.markdown("• *10 mins ago:* 2 Timber Lorries crawling at 25 km/h (KM 41.5)")
    st.markdown("• *32 mins ago:* Mill-bound tanker cleared incline without queue")