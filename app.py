import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from threading import Lock

load_dotenv()

# ---- Risk thresholds (days) ----
CRIT_MAX = int(os.getenv("CRITICAL_MAX_DAYS", 3))
HIGH_MAX = int(os.getenv("HIGH_MAX_DAYS", 7))
MED_MAX  = int(os.getenv("MEDIUM_MAX_DAYS", 14))
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Model cache to avoid disk loads per request
_model_cache = {}
_model_lock = Lock()

# Expected metric order (must align with training)
FEATURES = ["Temperature", "Vibration_Level", "Voltage", "Operating_Hours"]

def _normalize_asset_type(name: str) -> str:
    return (name or "").strip()

def list_models():
    """Return a dict {asset_type_display: path} for all trained models."""
    models = {}
    if not os.path.exists(MODEL_DIR):
        return models
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("regressor_") and fname.endswith(".pkl"):
            key = fname.replace("regressor_", "").replace(".pkl", "").replace("_", " ")
            models[key] = os.path.join(MODEL_DIR, fname)
    return models

def load_model(asset_type: str):
    """Dynamically load model for given asset type, with caching."""
    at = _normalize_asset_type(asset_type)
    available = list_models()
    if at not in available:
        raise ValueError(f"No model found for assetType '{at}'. Available: {list(available.keys())}")
    path = available[at]
    with _model_lock:
        if path in _model_cache:
            return _model_cache[path]
        model = joblib.load(path)
        _model_cache[path] = model
        return model

def risk_from_days(days: float) -> str:
    if days <= CRIT_MAX: return "Critical"
    if days <= HIGH_MAX: return "High"
    if days <= MED_MAX:  return "Medium"
    return "Low"

def _vector_from_metrics(m: dict) -> np.ndarray:
    """Build the feature vector in the trained order."""
    vals = []
    for key in FEATURES:
        try:
            vals.append(float(m.get(key, 0.0)))
        except (TypeError, ValueError):
            raise ValueError(f"Metric '{key}' must be numeric.")
    return np.array([vals], dtype=float)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/models", methods=["GET"])
def models():
    return jsonify({"models": sorted(list(list_models().keys()))}), 200

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    asset_type = payload.get("assetType")
    if not asset_type:
        return jsonify({"error": "assetType is required"}), 400

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return jsonify({"error": "metrics must be an object with numeric fields"}), 400

    try:
        model = load_model(asset_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        X = _vector_from_metrics(metrics)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    try:
        days = float(model.predict(X)[0])
    except Exception as e:
        return jsonify({"error": f"model prediction failed: {str(e)}"}), 500

    days = max(0.0, days)
    return jsonify({
        "assetType": _normalize_asset_type(asset_type),
        "riskLevel": risk_from_days(days),
        "remainingUsefulLifeHrs": round(days * 24.0, 2)
        # "explanations": []  # Optional future work (e.g., SHAP)
    }), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
