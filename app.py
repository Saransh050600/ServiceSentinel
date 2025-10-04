import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

CRIT_MAX = int(os.getenv("CRITICAL_MAX_DAYS", 3))
HIGH_MAX = int(os.getenv("HIGH_MAX_DAYS", 7))
MED_MAX = int(os.getenv("MEDIUM_MAX_DAYS", 14))

app = Flask(__name__)
model = joblib.load("models/regressor.pkl")  # trained pipeline (scaler + RF)

def bucket_risk(days):
    if days <= CRIT_MAX: return "Critical"
    if days <= HIGH_MAX: return "High"
    if days <= MED_MAX:  return "Medium"
    return "Low"

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    metrics = payload.get("metrics", {})
    # Expected keys as trained:
    temp = float(metrics.get("Temperature", 0.0))
    vib  = float(metrics.get("Vibration_Level", 0.0))
    volt = float(metrics.get("Voltage", 0.0))
    hrs  = float(metrics.get("Operating_Hours", 0.0))
    X = np.array([[temp, vib, volt, hrs]])
    days_pred = float(model.predict(X)[0])
    days_pred = max(0.0, days_pred)  # clamp
    rul_hours = days_pred * 24.0
    risk = bucket_risk(days_pred)
    return jsonify({
        "riskLevel": risk,
        "remainingUsefulLifeHrs": round(rul_hours, 2),
        "explanations": []  # You can add SHAP/feature effects later
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
