# app.py
# PURPOSE: Flask web application for HeartNet — real-time heart disease prediction.
#          Run with:  python app.py

import os, pickle
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

from model import HeartNet

app = Flask(__name__)

# ── LOAD MODEL ASSETS ────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_assets():
    """Load model, scaler, and feature names from disk once."""
    checkpoint    = torch.load('heart_model.pth', map_location=device)
    input_size    = checkpoint['input_size']
    test_accuracy = checkpoint.get('test_accuracy', 0)

    model = HeartNet(input_size=input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names, input_size, test_accuracy


model, scaler, feature_names, input_size, test_accuracy = load_assets()
print(f"Model loaded on {device}  |  Test Accuracy: {test_accuracy:.2f}%")
print(f"Feature names ({len(feature_names)}): {feature_names}")


def build_feature_vector(form_data: dict) -> np.ndarray:
    """
    Convert raw form values into the same feature vector used during training.

    Raw inputs from the web form:
        age, sex, trestbps, chol, fbs, thalch, exang, oldpeak,
        cp  (one of: typical angina | atypical angina | non-anginal | asymptomatic),
        restecg (one of: normal | lv hypertrophy | st-t abnormality)

    After one-hot expansion this matches feature_names exactly.
    """
    age      = float(form_data['age'])
    sex      = 1.0 if form_data['sex'] == 'Male' else 0.0
    trestbps = float(form_data['trestbps'])
    chol     = float(form_data['chol'])
    fbs      = 1.0 if form_data['fbs'] == '1' else 0.0
    thalch   = float(form_data['thalch'])
    exang    = 1.0 if form_data['exang'] == '1' else 0.0
    oldpeak  = float(form_data['oldpeak'])
    cp_val   = form_data['cp']
    restecg_val = form_data['restecg']

    # One-hot: cp
    cp_options = ['asymptomatic', 'atypical angina', 'non-anginal', 'typical angina']
    cp_vec = [1.0 if f'cp_{opt}' == f'cp_{cp_val}' else 0.0 for opt in cp_options]

    # One-hot: restecg
    restecg_options = ['lv hypertrophy', 'normal', 'st-t abnormality']
    restecg_vec = [1.0 if f'restecg_{opt}' == f'restecg_{restecg_val}' else 0.0
                   for opt in restecg_options]

    # Assemble in column order used during training
    vec_dict = {
        'age'     : age,
        'sex'     : sex,
        'trestbps': trestbps,
        'chol'    : chol,
        'fbs'     : fbs,
        'thalch'  : thalch,
        'exang'   : exang,
        'oldpeak' : oldpeak,
    }
    for opt in cp_options:
        vec_dict[f'cp_{opt}'] = 1.0 if cp_val == opt else 0.0
    for opt in restecg_options:
        vec_dict[f'restecg_{opt}'] = 1.0 if restecg_val == opt else 0.0

    # Build array aligned to feature_names
    vec = np.array([vec_dict.get(fn, 0.0) for fn in feature_names],
                   dtype=np.float32).reshape(1, -1)
    return vec


@app.route('/')
def index():
    return render_template('index.html', test_accuracy=f"{test_accuracy:.2f}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw = build_feature_vector(request.form)
        scaled = scaler.transform(raw).astype(np.float32)
        tensor = torch.tensor(scaled).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        no_disease_pct = round(float(probs[0]) * 100, 2)
        disease_pct    = round(float(probs[1]) * 100, 2)
        label          = 'Disease Detected' if probs[1] >= 0.5 else 'No Disease'

        return jsonify({
            'label'         : label,
            'disease_pct'   : disease_pct,
            'no_disease_pct': no_disease_pct,
            'risk_level'    : ('High' if disease_pct >= 70 else
                               'Moderate' if disease_pct >= 40 else 'Low'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
