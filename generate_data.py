import pandas as pd
import numpy as np

np.random.seed(42)
num_records = 5000

# 1. Simulate Contextual Variables
hours = np.random.randint(6, 20, num_records) # Daylight hours 6 AM - 8 PM
days = np.random.randint(0, 7, num_records) # 0 = Monday, 6 = Sunday
rain_intensity = np.random.choice([0, 1, 2], num_records, p=[0.7, 0.2, 0.1]) # 0=Clear, 1=Light, 2=Heavy

# 2. Domain Logic (The "Why" it slows down)
# High risk if: Peak mill hours (10-14), Weekdays (0-4), or Heavy Rain (2)
risk_scores = np.zeros(num_records)

for i in range(num_records):
    base_risk = 0.2
    if 10 <= hours[i] <= 14:
        base_risk += 0.4
    if days[i] < 5:
        base_risk += 0.2
    if rain_intensity[i] == 2:
        base_risk += 0.3
        
    risk_scores[i] = min(base_risk + np.random.uniform(-0.1, 0.1), 1.0)

# Convert to binary target: 1 = High Convoy Risk (Red), 0 = Low Risk (Green)
target = (risk_scores > 0.6).astype(int)

df = pd.DataFrame({
    'hour_of_day': hours,
    'day_of_week': days,
    'rain_intensity': rain_intensity,
    'high_risk_bottleneck': target
})

df.to_csv('synthetic_traffic_data.csv', index=False)