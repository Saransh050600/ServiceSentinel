import os, joblib, numpy as np
from flask import Flask, request, jsonify, abort
from dotenv import load_dotenv

load_dotenv()

CRIT_MAX = int(os.getenv("CRITICAL_MAX_DAYS", 3))
HIGH_MAX = int(os.getenv("HIGH_MAX_DAYS", 7))
MED_MAX  = int(os.getenv("MEDIUM_MAX_DAYS", 14))
API_KEY  = os.getenv("API_KEY")

MODEL_PATH = "models/regressor.pkl"

app = Flask(__name__)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found. Run train_model.py first.")
    return joblib.load(MODEL_PATH)

model = load_model()

def risk_from_days(days: float) -> str:
    if days <= CRIT_MAX: return "Critical"
    if days <= HIGH_MAX: return "High"
    if days <= MED_MAX:  return "Medium"
    return "Low"

def check_key(req):
    if API_KEY:
        key = req.headers.get("x-api-key")
        if key != API_KEY:
            abort(401)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    check_key(request)
    p = request.get_json(force=True) or {}
    m = p.get("metrics", {})
    # Expected metrics (names match Apex call)
    X = np.array([[float(m.get("Temperature", 0.0)),
                   float(m.get("Vibration_Level", 0.0)),
                   float(m.get("Voltage", 0.0)),
                   float(m.get("Operating_Hours", 0.0))]])
    days = float(model.predict(X)[0])
    days = max(0.0, days)
    return jsonify({
        "riskLevel": risk_from_days(days),
        "remainingUsefulLifeHrs": round(days * 24.0, 2),
        "explanations": []  # extend with SHAP later if needed
    })

# OPTIONAL: retrain on demand (requires API_KEY) — runs train_model.py then reloads the model
@app.route("/train", methods=["POST"])
def train():
    check_key(request)
    # Train in-process by invoking the trainer script
    # Note: On Render free plans, long jobs might time out — consider a cron or background worker for production.
    from subprocess import run, CalledProcessError
    try:
        r = run(["python", "train_model.py"], capture_output=True, text=True, check=True)
        global model
        model = load_model()
        return jsonify({"status": "trained", "stdout": r.stdout[-1000:]}), 200
    except CalledProcessError as e:
        return jsonify({"status": "error", "stderr": e.stderr[-2000:]}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
