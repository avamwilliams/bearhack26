import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

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
X_train, y_train = smote.fit_resample(X_train, y_train)
#2=normal, 1=spoofing, 0=data altercation
print(f"{pd.Series(y_train).value_counts().to_dict()}")



#training the random forest
rf = RandomForestClassifier(
    n_estimators=200, #num of trees
    max_depth=None,   #trees can fully grow, won't get cut off
    min_samples_split=2,
    random_state=67,
    n_jobs=-1       #uses all CPU cores
)

rf.fit(X_train, y_train)

#evaluation
y_pred = rf.predict(X_test)

#how accurate the predictions were 
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_encoder.classes_, ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix (3-Class)")
importances = pd.Series(rf.feature_importances_, index=features.columns)
top20 = importances.nlargest(20).sort_values()

colors = ["#036f2c" if f in bio_columns else "#d78602" for f in top20.index]
top20.plot(kind='barh', color=colors, ax=axes[1])
axes[1].set_title("Top 20 Features in Feature Selection (Biometric=Green, Network=Orange)")
axes[1].set_xlabel("Importance")
plt.tight_layout()
plt.show()


#pickles -> convert python objects to bytes so we can load it later
#trains model saves it to a pickle so we can load it later for inference
#Joblib is a library built on top of pickle where it actually compresses it and loads it
joblib.dump(rf, "rf_model.pkl")

