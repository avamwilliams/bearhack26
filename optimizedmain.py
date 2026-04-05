import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import shap
from tensorflow import keras
import io

app = Flask(__name__)
CORS(app)

df = pd.read_csv("wustl-ehms-2020_with_attacks_categories.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=['SrcMac', 'DstMac', 'Dir', 'SrcAddr', 'DstAddr', 'Sport', 'Label'])
bio_columns  = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST']
label_column = 'Attack Category'

df = pd.get_dummies(df, columns=['Flgs'], drop_first=False)
feature_cols = [col for col in df.columns if col != label_column]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[feature_cols] = df[feature_cols].fillna(0)

network_columns = [col for col in df.columns if col not in bio_columns + [label_column]]
all_cols        = network_columns + bio_columns

label_encoder  = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df[label_column].str.strip())

features        = df[all_cols]
scaler          = StandardScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    scaled_features, encoded_labels, test_size=0.2, random_state=67, stratify=encoded_labels
)
smote = SMOTE(random_state=67, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_raw[all_cols], y_train_raw)

model_comb = RandomForestClassifier(n_estimators=100, random_state=67, n_jobs=-1)
model_comb.fit(X_train_bal, y_train_bal)

explainer = shap.TreeExplainer(model_comb)

normal_mask   = encoded_labels == 2
X_normal_only = scaled_features[normal_mask][all_cols]

isoforest = IsolationForest(contamination=0.125, random_state=67, n_jobs=-1)
isoforest.fit(X_normal_only)
iso_full = isoforest.decision_function(scaled_features[all_cols])

n_features  = len(all_cols)
autoencoder = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(n_features, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_normal_only.values, X_normal_only.values,
                epochs=30, batch_size=64, validation_split=0.1, verbose=0)

ae_full = np.mean(
    (scaled_features[all_cols].values - autoencoder.predict(scaled_features[all_cols].values, verbose=0)) ** 2,
    axis=1
)
ae_p5  = np.percentile(ae_full, 5)
ae_p99 = np.percentile(ae_full, 99)


def generate_suggestions(prediction, alert_level, top5_shap_keys):
    if prediction.lower() == "normal" and alert_level == "LOW":
        return ["No action needed: traffic appears normal."]

    suggestions = set()

    if alert_level in ("MEDIUM", "HIGH") and prediction.lower() == "normal":
        suggestions.add("Model predicts normal but anomaly signals are elevated: manual review recommended.")

    for f in top5_shap_keys:
        if "Load" in f:
            suggestions.add("High traffic load detected: consider rate limiting.")
        elif "Pkt" in f or "pkt" in f:
            suggestions.add("Unusual packet behavior: inspect packet frequency and size.")
        elif "Jitter" in f:
            suggestions.add("Network instability detected: check for spoofing or interference.")
        elif "Temp" in f or "Heart_rate" in f or "SpO2" in f or "Pulse" in f:
            suggestions.add("Abnormal biometric readings: verify device integrity.")
        elif "Loss" in f:
            suggestions.add("Packet loss observed: investigate possible tampering.")
        elif "SYS" in f or "DIA" in f:
            suggestions.add("Abnormal blood pressure readings: check sensor calibration.")
        elif "Resp" in f or "ST" in f:
            suggestions.add("Abnormal respiratory/ECG readings: verify patient sensor connections.")
        elif "Flgs" in f:
            suggestions.add("Unusual network flags detected: inspect packet headers.")
        else:
            suggestions.add(f"Investigate anomalous feature: {f}")

    return list(suggestions)

def analyze_sample_json(row_idx, scaled_df, raw_df):
    sample_scaled  = scaled_df.iloc[[row_idx]]
    sample_np      = sample_scaled.values

    rf_probs       = model_comb.predict_proba(sample_scaled)[0]
    rf_pred_idx    = rf_probs.argmax()
    rf_pred_label  = label_encoder.inverse_transform([rf_pred_idx])[0]
    rf_attack_prob = 1 - rf_probs[2]

    iso_score_raw  = isoforest.decision_function(sample_scaled)[0]
    iso_score_norm = float(np.clip(
        1 - (iso_score_raw - iso_full.min()) / (iso_full.max() - iso_full.min()), 0, 1
    ))

    reconstructed  = autoencoder.predict(sample_np, verbose=0)
    per_feat_error = (sample_np[0] - reconstructed[0]) ** 2
    ae_error_total = per_feat_error.mean()
    ae_norm        = float(np.clip((ae_error_total - ae_p5) / (ae_p99 - ae_p5), 0, 1))

    severity    = float(np.clip(
        (0.5 * rf_attack_prob + 0.3 * iso_score_norm + 0.2 * ae_norm) * 100, 0, 100
    ))
    alert_level = "LOW" if severity <= 30 else "MEDIUM" if severity <= 65 else "HIGH"

    shap_vals     = explainer.shap_values(sample_scaled)
    shap_for_pred = shap_vals[rf_pred_idx][0] if isinstance(shap_vals, list) else shap_vals[0, :, rf_pred_idx]
    top5_shap     = pd.Series(np.abs(shap_for_pred), index=all_cols).nlargest(5)
    top5_recon    = pd.Series(per_feat_error, index=all_cols).nlargest(5)

    suggestions = generate_suggestions(rf_pred_label, alert_level, list(top5_shap.index))
    return {
        "sample": row_idx + 1,
        "prediction": rf_pred_label,
        "confidence": round(float(rf_probs.max()) * 100, 1),
        "severity": round(severity, 1),
        "alert_level": alert_level,
        "class_probs": {
            "normal":   round(float(rf_probs[2]) * 100, 1),
            "spoofing": round(float(rf_probs[1]) * 100, 1),
            "data_alt": round(float(rf_probs[0]) * 100, 1)
        },
        "signals": {
            "rf_attack_prob": round(rf_attack_prob * 100, 1),
            "iso_score":      round(iso_score_norm * 100, 1),
            "ae_anomaly":     round(ae_norm * 100, 1)
        },
        "top5_shap":  {k: round(float(v), 4) for k, v in top5_shap.items()},
        "top5_recon": {k: round(float(v), 4) for k, v in top5_recon.items()},
        "suggestions": suggestions
    }


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file       = request.files['file']
    test_input = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    test_input.columns = test_input.columns.str.strip()
    test_input = pd.get_dummies(test_input, columns=['Flgs'], drop_first=False)

    for col in all_cols:
        if col in test_input.columns:
            test_input[col] = pd.to_numeric(test_input[col], errors='coerce')
        else:
            test_input[col] = 0.0
    test_input[all_cols] = test_input[all_cols].fillna(0)

    test_scaled = pd.DataFrame(scaler.transform(test_input[all_cols]), columns=all_cols)

    results = [analyze_sample_json(i, test_scaled, test_input) for i in range(len(test_scaled))]
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, port=5000)