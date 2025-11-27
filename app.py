import matplotlib
matplotlib.use('Agg') # Fix for main thread error
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, jsonify
import os
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import shap
import io
import base64
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- DATABASE CONFIG ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    prediction = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    mean_radius = db.Column(db.Float)
    mean_area = db.Column(db.Float)

with app.app_context():
    db.create_all()

# --- MODEL LOADING ---
MODEL_PATH = os.path.join("models", "breast_cancer_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

try:
    explainer = shap.TreeExplainer(model)
except:
    explainer = None

FEATURE_NAMES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
    'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
]

@app.route("/")
def home():
    return render_template("index.html")

# --- SINGLE PREDICTION ROUTE ---
@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()["data"]
    else:
        data = [float(request.form.get(f)) for f in FEATURE_NAMES]

    try:
        arr_original = np.array(data).reshape(1, -1)
        arr_scaled = scaler.transform(arr_original)
        
        pred = int(model.predict(arr_scaled)[0])
        prob_malignant = float(model.predict_proba(arr_scaled)[0][1])
        
        # LOGIC FIX: Confidence should match the label
        if pred == 1:
            label = "Malignant"
            confidence = prob_malignant
        else:
            label = "Benign"
            confidence = 1.0 - prob_malignant # Invert for Benign

        # Save to DB
        new_record = Patient(
            prediction=label,
            confidence=round(confidence * 100, 2),
            mean_radius=data[0],
            mean_area=data[3]
        )
        db.session.add(new_record)
        db.session.commit()

        # Generate SHAP
        explanation_image = None
        if explainer:
            shap_values = explainer.shap_values(arr_scaled)
            # Handle list output for classifiers
            if isinstance(shap_values, list):
                sv = shap_values[1][0] 
                base = explainer.expected_value[1]
            else:
                sv = shap_values[0]
                base = explainer.expected_value
            
            shap_exp = shap.Explanation(values=sv, base_values=base, data=arr_original[0], feature_names=FEATURE_NAMES)
            
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(shap_exp, max_display=8, show=False)
            plt.title(f"Factors driving '{label}' diagnosis", fontsize=12)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            explanation_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

        return jsonify({
            "prediction": pred,
            "probability": confidence, # Send fixed confidence
            "label": label,
            "explanation_image": explanation_image
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- CSV BATCH PREDICTION ROUTE ---
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        
        df = pd.read_csv(file)
        
        # Validation
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing: return jsonify({"error": f"Missing columns: {missing}"}), 400

        X = df[FEATURE_NAMES]
        X_scaled = scaler.transform(X)
        
        preds = model.predict(X_scaled)
        probs_malignant = model.predict_proba(X_scaled)[:, 1]
        
        results_label = []
        results_conf = []
        
        for p, prob in zip(preds, probs_malignant):
            if p == 1:
                results_label.append("Malignant")
                # Format as nice percentage string
                results_conf.append(f"{round(prob * 100, 2)}%")
            else:
                results_label.append("Benign")
                # Invert probability for Benign cases
                results_conf.append(f"{round((1 - prob) * 100, 2)}%")
        
        df['AI_Prediction'] = results_label
        df['Confidence_Score'] = results_conf
        
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return base64.b64encode(output.getvalue()).decode('utf-8')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- HISTORY ROUTE ---
@app.route("/api/history", methods=["GET"])
def get_history():
    records = Patient.query.order_by(Patient.timestamp.desc()).limit(10).all()
    data = []
    for r in records:
        data.append({
            "id": r.id,
            "date": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "prediction": r.prediction,
            "confidence": r.confidence,
            "radius": round(r.mean_radius, 2),
            "area": round(r.mean_area, 2)
        })
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)