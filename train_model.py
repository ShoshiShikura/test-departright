import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the synthetic data
df = pd.read_csv('synthetic_traffic_data.csv')

X = df[['hour_of_day', 'day_of_week', 'rain_intensity']]
y = df['high_risk_bottleneck']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save the trained model for the web app
with open('sawit_model.pkl', 'wb') as f:
    pickle.dump(model, f)