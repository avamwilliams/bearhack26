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