from sklearn.ensemble import IsolationForest
import numpy as np

# Train model (dummy or real data)
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Dummy training (must match feature size = 3)
X_dummy = np.random.rand(200, 3)
model.fit(X_dummy)

def run_model(input_data):
    """
    input_data: list or array of 3 sensor values
    """
    x = np.array(input_data).reshape(1, -1)
    pred = model.predict(x)[0]

    if pred == -1:
        return "⚠️ Anomaly Detected"
    else:
        return "✅ Normal"