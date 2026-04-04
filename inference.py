import pandas as pd
import joblib

#load saved objects
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

def preprocess_input(df):
    df.columns = df.columns.str.strip()

    #same drops as training
    drop = ['SrcMac', 'DstMac', 'Dir', 'SrcAddr', 'DstAddr', 'Sport', 'Label']
    df = df.drop(columns=[col for col in drop if col in df.columns], errors='ignore')

    #one hot encode flags
    df = pd.get_dummies(df, columns=['Flgs'], drop_first=False)

    #make sure expected columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    #ensure column order
    df = df[feature_columns]

    #numeric conversions
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    #scaling
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )

    return df_scaled


def predict(file_path):
    df = pd.read_csv(file_path)

    X = preprocess_input(df)
    preds = rf.predict(X)

    #convert back to labels
    decoded = label_encoder.inverse_transform(preds)

    return decoded

