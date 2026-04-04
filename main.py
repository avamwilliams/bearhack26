import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("wustl-ehms-2020_with_attacks_categories.csv")
df.columns = df.columns.str.strip()
print("Shape:", df.shape)

drop = ['SrcMac', 'DstMac', 'Dir', 'SrcAddr', 'DstAddr', 'Sport', 'Label']
df = df.drop(columns=drop)
bio_columns = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST']
label_column = 'Attack Category'

print(f"Bio features:{len(bio_columns)}")

#one hot encoding flags + changing numeric columns 
df = pd.get_dummies(df, columns=['Flgs'], drop_first=False)
feature_cols = [col for col in df.columns if col != label_column]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[feature_cols] = df[feature_cols].fillna(0)
network_columns = [col for col in df.columns if col not in bio_columns + [label_column]]
print(f"Network features:{len(network_columns)}")

#one hot for labels + finishing full feature set
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df[label_column].str.strip())
print("\nClass mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
features = df[network_columns + bio_columns]
scaler = StandardScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

#SMOTE needed for balancing data for random fores
X_train, X_test, y_train, y_test = train_test_split(scaled_features, encoded_labels, test_size=0.2, random_state=67, stratify=encoded_labels)
smote = SMOTE(random_state=67, k_neighbors=5)
X_train, y_train_SMOTE = smote.fit_resample(X_train, y_train)
#2=normal, 1=spoofing, 0=data altercation
print(f"{pd.Series(y_train_SMOTE).value_counts().to_dict()}")

