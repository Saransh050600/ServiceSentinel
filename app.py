import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from threading import Lock

load_dotenv()

# --- Risk thresholds in DAYS (can be overridden via env) ---
CRIT_MAX = int(os.getenv("CRITICAL_MAX_DAYS", 3))
HIGH_MAX = int(os.getenv("HIGH_MAX_DAYS", 7))
MED_MAX  = int(os.getenv("MEDIUM_MAX_DAYS", 14))

# --- Where trained models live (align with training script default: models_eval) ---
MODEL_DIR = os.getenv("MODEL_DIR", "models_eval")

# Fallback for very old models that don't have .meta.json
FEATURES_FALLBACK = ["Temperature", "Vibration_Level", "Voltage", "Operating_Hours"]

# Cache: per assetType → {"model": pipeline, "features": [feature names]}
_model_cache = {}
_model_lock = Lock()


# ----------------- Helpers -----------------
def _normalize_asset_type(name: str) -> str:
    """Normalize assetType string for consistent matching."""
    return (name or "").strip()


def list_models():
    """
    Scan MODEL_DIR for regressor_*.pkl files.
    For each model, try to load its meta JSON to get:
      - assetType (canonical key)
      - features (dynamic feature names used in training)
    Returns: { assetType: {"model_path": "...", "features": [...] } }
    """
    models = {}
    if not os.path.exists(MODEL_DIR):
        return models

    for fname in os.listdir(MODEL_DIR):
        if not (fname.startswith("regressor_") and fname.endswith(".pkl")):
            continue

        model_path = os.path.join(MODEL_DIR, fname)
        base = fname[:-4]  # drop ".pkl"
        meta_path = os.path.join(MODEL_DIR, base + ".meta.json")

        asset_type_key = None
        features = None

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                # Prefer assetType from meta
                asset_type_key = _normalize_asset_type(meta.get("assetType"))
                features = meta.get("features") or None
            except Exception:
                # If meta is corrupt, fall back to filename
                pass

        if not asset_type_key:
            # Derive assetType from filename: regressor_Motor_Type_A → "Motor Type A"
            asset_type_key = base.replace("regressor_", "").replace("_", " ")
            asset_type_key = _normalize_asset_type(asset_type_key)

        if not features:
            # For old models (from older training code) that didn't save features
            features = FEATURES_FALLBACK

        models[asset_type_key] = {
            "model_path": model_path,
            "features": features
        }

    return models


def load_model(asset_type: str):
    """
    Load (and cache) the model + features for a given assetType.
    Returns: {"model": pipeline, "features": [feature names]}
    """
    at = _normalize_asset_type(asset_type)
    available = list_models()
    if at not in available:
        raise ValueError(f"No model found for assetType '{at}'. Available: {list(available.keys())}")

    info = available[at]
    model_path = info["model_path"]
    features = info["features"]

    if not features:
        raise ValueError(f"No feature metadata found for assetType '{at}'.")

    with _model_lock:
        if at in _model_cache:
            return _model_cache[at]

        model = joblib.load(model_path)
        _model_cache[at] = {"model": model, "features": features}
        return _model_cache[at]


def risk_from_days(days: float) -> str:
    """Map remaining days to risk bucket."""
    if days <= CRIT_MAX:
        return "Critical"
    if days <= HIGH_MAX:
        return "High"
    if days <= MED_MAX:
        return "Medium"
    return "Low"


def _vector_from_metrics(m: dict, features: list) -> np.ndarray:
    """
    Build a 2D feature vector [1 x n_features] in the SAME ORDER
    as used during training (from meta['features']).
    """
    vals = []
    for key in features:
        try:
            vals.append(float(m.get(key, 0.0)))
        except (TypeError, ValueError):
            raise ValueError(f"Metric '{key}' must be numeric (got: {m.get(key)!r}).")
    return np.array([vals], dtype=float)


# ----------------- Flask app -----------------
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/models", methods=["GET"])
def models():
    """
    Return list of assetTypes for which we have models.
    """
    return jsonify({"models": sorted(list(list_models().keys()))}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected payload:
    {
      "assetType": "Chiller Type A",
      "metrics": {
          "<feature1>": 123,
          "<feature2>": 45.6,
          ...
      }
    }

    Feature names MUST match those stored in the model's meta.json under "features".
    """
    payload = request.get_json(force=True) or {}
    asset_type = payload.get("assetType")
    if not asset_type:
        return jsonify({"error": "assetType is required"}), 400

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return jsonify({"error": "metrics must be an object with numeric fields"}), 400

    try:
        info = load_model(asset_type)
        model = info["model"]
        features = info["features"]

        X = _vector_from_metrics(metrics, features)
        days = float(model.predict(X)[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    days = max(0.0, days)
    return jsonify({
        "assetType": _normalize_asset_type(asset_type),
        "riskLevel": risk_from_days(days),
        "remainingUsefulLifeHrs": round(days * 24.0, 2),
        "remainingUsefulLifeDays": round(days, 2)
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
