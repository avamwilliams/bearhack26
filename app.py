

import os os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' import warnings warnings.filterwarnings('ignore') import logging logging.getLogger('tensorflow').setLevel(logging.ERROR) import pandas as pd import numpy as np import matplotlib matplotlib.use('Agg

pasted

how do i connect backend to my front end site

this needs to appear in my box
══════════════════════════════════════════════════════════════ IOMT SECURITY REPORT - Input sample #6 ══════════════════════════════════════════════════════════════ Prediction: Data Alteration Confidence: 100.0% Class probs: normal=0.0% spoofing=0.0% dataalt=100.0% ────────────────────────────────────────────────────────────── SEVERITY SCORE: 79.4 / 100 ALERT LEVEL: HIGH ────────────────────────────────────────────────────────────── Signal breakdown: RF attack probability : 100.0% (weight 50%) Isolation Forest score : 81.2% (weight 30%) Autoencoder anomaly : 25.0% (weight 20%) ────────────────────────────────────────────────────────────── Top 5 features used in classification (Data Alteration) Flgs M SHAP=0.1535 value=True DstJitter SHAP=0.1254 value=50.401 SrcLoad SHAP=0.0747 value=19026.0 DIntPkt SHAP=0.0670 value=50.406 SIntPkt SHAP=0.0517 value=52.139668 ────────────────────────────────────────────────────────────── Top 5 features deviating from normal (autoencoder) Flgs_ M reconerr=19.1422 value=True Flgs e recon_err=16.3703 value=False Packet_num recon_err=3.5717 value=109 SYS recon_err=2.5157 value=138 (biometric) Load recon_err=2.4456 value=25368.0 ══════════════════════════════════════════════════════════════

appear in this front end
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">
  <style>
* {
  box-sizing: border-box;
}
html, body {
  margin: 0;
  height: 100%;
}
body {
  overflow-x: hidden;
  background: url('https://i.postimg.cc/d1Ms2R3J/Screenshot-2026-04-04-103604.png') center/cover no-repeat;
  font-family: 'Inter', sans-serif;
}
/* HEADER */
.header {
  width: 100%;
  height: 100px;
  background-color: 
#1d2040;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
}
.logo {
  font-weight: bold;
  font-size: 40px;
  color: white;
  margin: 0;
}
.nav-buttons {
  display: flex;
  gap: 10px;
}
.button {
  border: 1px solid white;
  border-radius: 6px;
  color: white;
  padding: 12px 20px;
  font-size: 18px;
  cursor: pointer;
  background-color: transparent;
  transition: 0.3s;
  text-decoration: none;
}
.button:hover {
  background-color: white;
  color: black;
  text-decoration: none;
}
/* TITLES */
.title {
  color: white;
  font-size: 40px;
  font-weight: bold;
  text-align: center;
  margin-top: 40px;
}
.center-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  margin-top: 30px;
}
/* BOXES */
.fancybox {
  width: 40%;
  padding: 20px;
  background: rgba(29, 32, 64, 0.6);
  border-radius: 12px;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
/* TEXT */
.subtitle {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 20px;
  color: white;
}
.pill {
  background-color: white;
  color: 
#1d2040;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 16px;
  font-weight: bold;
}
/* FILE BUTTON */
.file-button {
  display: inline-block;
  background-color: white;
  color: 
#1d2040;
  padding: 10px 18px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: bold;
  cursor: pointer;
  transition: 0.3s;
  margin-top: 10px;
}
.file-button:hover {
  background-color: 
#e6e6e6;
}
#fileName {
  color: white;
  text-align: center;
  margin-top: 10px;
}
  </style>
</head>
<body>
  <div class="header">
    <p class="logo">HealthHop</p>
<div class="nav-buttons">
  <a href="import.html" class="button">Import</a>
  <a href="analysis.html" class="button">Analysis</a>
  <a href="info.html" class="button">Info</a>
</div>
  </div>

  <div class="title">Analysing Health Data</div>
  <div class="center-container">
    <div class="fancybox">
      <div class="subtitle">
        Analysing data for:
        <span class="pill">Patient</span>
      </div>
    </div>

    <div class= "fancybox" style="width:70%">
    <div class = "subtitle">
    The imported data indicates that the medical equipment is performing normally. <br><br>The imported data indicates that the medical equipment is performing abnormally. This suggests the device is compromised.
    </div>
    </div>
<script>
  const input = document.getElementById("fileUpload");
  const fileName = document.getElementById("fileName");
  input.addEventListener("change", () => {
    fileName.textContent = input.files[0]?.name || "";
  });
</script>
</body>
</html>

heres the backend were using
suggestions.py
import pandas as pd
import numpy as np
import joblib
def generate_suggestions(file_path):
    #load everything
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    #load data
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    #same preprocessing as training
    drop_cols = ['SrcMac', 'DstMac', 'Dir', 'SrcAddr', 'DstAddr', 'Sport', 'Label']
    df = df.drop(columns=[col for col in drop_cols if col in df], errors='ignore')
    if 'Flgs' in df.columns:
        df = pd.get_dummies(df, columns=['Flgs'], drop_first=False)
    #align features
    for col in feature_columns:
        if col not in df:
            df[col] = 0
    df = df[feature_columns]
    #scale
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )
    #predict
    preds = model.predict(df_scaled)
    pred_labels = label_encoder.inverse_transform(preds)
    #feature importance
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    suggestions = []
    for i, row in df.iterrows():
        label = pred_labels[i]
        if label.lower() == "normal":
            suggestions.append("No action needed — traffic appears normal.")
            continue
        #get top contributing features for this row
        row_values = row.abs()
        contribution = importances * row_values
        top_features = contribution.nlargest(5).index.tolist()
        #generate suggestions
        row_suggestions = []
        for f in top_features:
            if "Load" in f:
                row_suggestions.append("High traffic load detected: consider rate limiting.")
            elif "Pkt" in f:
                row_suggestions.append("Unusual packet behavior: inspect packet frequency and size.")
            elif "Jitter" in f:
                row_suggestions.append("Network instability detected: check for spoofing or interference.")
            elif "Temp" in f or "Heart_rate" in f:
                row_suggestions.append("Abnormal biometric readings: verify device integrity.")
            elif "Loss" in f:
                row_suggestions.append("Packet loss observed: investigate possible tampering.")
            else:
                row_suggestions.append(f"Check feature: {f}")
        suggestions.append({
            "prediction": label,
            "top_features": top_features,
            "suggestions": list(set(row_suggestions))  #remove duplicates
        })
    return suggestions

and testsuggestions .py

from suggestions import generate_suggestions
#generate suggestions
results = generate_suggestions("normal.csv")
print(results)
#compromised payload
comp_result = generate_suggestions("compromised.csv")
print(comp_result)

Show more
9:43 PM
Looking at your setup, the cleanest approach is to build a small Flask API that runs your Python backend and serves results to your HTML frontend via a fetch call.

Here's the plan:

1. Create app.py (Flask Server)

Flask backend API
Flask backend API
2. Updated analysis.html

Created a file, read a file
Created a file, read a file
Here's how the whole thing connects:

Install the two new dependencies first:

bash
pip install flask flask-cors
Then start the backend:

bash
python app.py
How it works end-to-end:

The user opens analysis.html in a browser and picks a CSV file
Clicking Run Analysis sends the file to http://localhost:5000/analyze via a POST request
app.py receives it, saves it to a temp file, calls your existing generate_suggestions(), and returns the results as JSON
The frontend formats and displays the report in the styled box — with colour-coded HIGH / MEDIUM / LOW alert badges
Important: app.py needs to live in the same folder as suggestions.py and all your .pkl model files, since it imports from suggestions directly.

If you later deploy the backend to a server (e.g. Render, Railway), just change this one line in the HTML:

js
const API_URL = "http://localhost:5000/analyze";  // swap for your deployed URL
Analysis
Code · HTML 
App
PY 

Want to be notified when Claude responds?




Claude is AI and can make mistakes. Please double-check responses.
App · PY
Copy

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os
from suggestions import generate_suggestions
 
app = Flask(__name__)
CORS(app)  # allows your HTML page to call this API
 
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
 
    uploaded = request.files["file"]
    # save to a temp file so generate_suggestions can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name
 
    try:
        results = generate_suggestions(tmp_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)
 

