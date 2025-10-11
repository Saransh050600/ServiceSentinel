import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# ---- Risk thresholds (days) ----
CRIT_MAX = int(os.getenv("CRITICAL_MAX_DAYS", 3))
HIGH_MAX = int(os.getenv("HIGH_MAX_DAYS", 7))
MED_MAX  = int(os.getenv("MEDIUM_MAX_DAYS", 14))
MODEL_DIR = "models"

def list_models():
    """Return a dict {asset_type: path} for all trained models."""
    models = {}
    if not os.path.exists(MODEL_DIR):
        return models
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("regressor_") and fname.endswith(".pkl"):
            asset_type = fname.replace("regressor_", "").replace(".pkl", "").replace("_", " ")
            models[asset_type] = os.path.join(MODEL_DIR, fname)
    return models

def load_model(asset_type: str):
    """Dynamically load model for given asset type."""
    models = list_models()
    if asset_type not in models:
        raise ValueError(f"No model found for assetType '{asset_type}'. Available: {list(models.keys())}")
    return joblib.load(models[asset_type])

def risk_from_days(days: float) -> str:
    if days <= CRIT_MAX: return "Critical"
    if days <= HIGH_MAX: return "High"
    if days <= MED_MAX:  return "Medium"
    return "Low"

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    asset_type = payload.get("assetType")
    if not asset_type:
        return jsonify({"error": "assetType is required"}), 400

    try:
        model = load_model(asset_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    m = payload.get("metrics", {})
    X = np.array([[
        float(m.get("Temperature", 0.0)),
        float(m.get("Vibration_Level", 0.0)),
        float(m.get("Voltage", 0.0)),
        float(m.get("Operating_Hours", 0.0)),
    ]])

    days = float(model.predict(X)[0])
    days = max(0.0, days)

    return jsonify({
        "assetType": asset_type,
        "riskLevel": risk_from_days(days),
        "remainingUsefulLifeHrs": round(days * 24.0, 2)
    }), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
