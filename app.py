from flask import Flask, request, jsonify
import joblib
import os

# --- 1. Load Models and Encoders ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
clf = joblib.load(os.path.join(MODEL_DIR, 'asset_risk_model.pkl'))
rul_reg = joblib.load(os.path.join(MODEL_DIR, 'rul_model.pkl'))
asset_encoder = joblib.load(os.path.join(MODEL_DIR, 'asset_encoder.pkl'))
risk_encoder = joblib.load(os.path.join(MODEL_DIR, 'risk_encoder.pkl'))

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Service Sentinel AI Model is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts both Risk Level (classification) and Remaining Useful Life Hours (regression).
    Expected JSON:
    {
        "AssetType": "Transformer",
        "Current": 10.5,
        "Flow_Rate": 25,
        "Frequency": 60,
        "Health_Score": 72,
        "Load": 80,
        "Oil_Level": 50,
        "Operating_Hours": 3400,
        "Pressure": 101.3,
        "RPM": 1200,
        "Steam_Flow": 0,
        "Temperature": 55,
        "Vibration": 2.1,
        "Voltage": 220
    }
    """
    try:
        data = request.get_json()
        asset_type = data.get('AssetType')
        if not asset_type:
            return jsonify({"error": "AssetType is required"}), 400

        features = [
            asset_encoder.transform([asset_type])[0],
            data.get('Current', 0),
            data.get('Flow_Rate', 0),
            data.get('Frequency', 0),
            data.get('Health_Score', 0),
            data.get('Load', 0),
            data.get('Oil_Level', 0),
            data.get('Operating_Hours', 0),
            data.get('Pressure', 0),
            data.get('RPM', 0),
            data.get('Steam_Flow', 0),
            data.get('Temperature', 0),
            data.get('Vibration', 0),
            data.get('Voltage', 0)
        ]

        # Predict Risk Level
        risk_pred = clf.predict([features])[0]
        risk_label = risk_encoder.inverse_transform([risk_pred])[0]

        # Predict Remaining Useful Life Hours
        rul_pred = rul_reg.predict([features])[0]

        return jsonify({
            "predicted_risk_level": risk_label,
            "predicted_remaining_life_hours": round(float(rul_pred), 2),
            "input_used": data
        })

    except ValueError as ve:
        return jsonify({"error": f"Encoding error: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
