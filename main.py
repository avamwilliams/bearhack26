import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import shap
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow import keras

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

#random forest 
random_for = RandomForestClassifier(n_estimators=100, random_state=67)
random_for.fit(X_train, y_train)
y_predictions = random_for.predict(X_test)

#findings for classification of random forest + confusion matrix to show what each was predicted vs reality
print("\nClassification Report")
print(classification_report(y_test, y_predictions, target_names=label_encoder.classes_))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_predictions, display_labels=label_encoder.classes_, ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix (3-Class)")
importances = pd.Series(random_for.feature_importances_, index=features.columns)
top20 = importances.nlargest(20).sort_values()

#shows most important features within features selection
colors = ["#036f2c" if f in bio_columns else "#d78602" for f in top20.index]
top20.plot(kind='barh', color=colors, ax=axes[1])
axes[1].set_title("Top 20 Features in Feature Selection (Biometric=Green, Network=Orange)")
axes[1].set_xlabel("Importance")
plt.tight_layout()
plt.savefig("results.png", dpi=150)
#plt.show()

#evaluation for checking only subsectiosn of features vs all of them combined 
def train_evaluate(X_train, y_train, X_test, y_test, testing_title):
    model = RandomForestClassifier(n_estimators=100, random_state=67, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)

    # one vs rest AUROC for multiclass labels (treats one label as positive and all others as negative)
    y_test_binary = label_binarize(y_test, classes=[0, 1, 2])
    auroc = roc_auc_score(y_test_binary, y_probability, average='macro', multi_class='ovr')
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    print(f"\nTest Name: {testing_title}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Spoofing F1: {report['Spoofing']['f1-score']:.4f}")
    return auroc, report, model


#re-split before smote for clean subsets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(scaled_features, encoded_labels, test_size=0.2, random_state=67, stratify=encoded_labels)

#using smote to make equal sized data again and spl
def get_smote_split(cols):
    X_train = X_train_raw[cols]
    X_test = X_test_raw[cols]
    sm = SMOTE(random_state=67, k_neighbors=5)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train_raw)
    return X_train_balanced, y_train_balanced, X_test, y_test_raw

#running auroc on each single item
auroc_network, report_net, model_net = train_evaluate(*get_smote_split(network_columns), "Network only features")
auroc_bio, report_bio, model_bio = train_evaluate(*get_smote_split(bio_columns), "Biometric only features")
auroc_combined, report_comb, model_comb = train_evaluate(*get_smote_split(network_columns + bio_columns), "Combined features")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = ['Network\nOnly', 'Biometric\nOnly', 'Combined']
aurocs = [auroc_network, auroc_bio, auroc_combined]
spoof_f1s = [report_net['Spoofing']['f1-score'], report_bio['Spoofing']['f1-score'], report_comb['Spoofing']['f1-score']]

#auroc comparison graph
bars = axes[0].bar(models, aurocs, color=['#d78602', '#036f2c', '#2c5fd7'], width=0.5)
axes[0].set_ylim(0.5, 1.05)
axes[0].set_title("AUROC by Feature Group")
axes[0].set_ylabel("AUROC (Macro One vs Rest)")
for bar, val in zip(bars, aurocs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha='center', fontweight='bold')

#spoofing F1 comparison
bars2 = axes[1].bar(models, spoof_f1s, color=['#d78602', '#036f2c', '#2c5fd7'], width=0.5)
axes[1].set_ylim(0, 1.0)
axes[1].set_title("Spoofing Detection F1 by Feature Group")
axes[1].set_ylabel("F1 Score")
for bar, val in zip(bars2, spoof_f1s):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha='center', fontweight='bold')
plt.suptitle("Does Adding Biometrics Improve Attack Detection?", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
#plt.show()

#shap contribution graph
X_test_combined = X_test_raw[network_columns + bio_columns]
explainer = shap.TreeExplainer(model_comb)
shap_values = explainer.shap_values(X_test_combined.iloc[:500])


if isinstance(shap_values, list):
    spoofing_shap = shap_values[1]
else:
    spoofing_shap = shap_values[:, :, 1]

plt.figure()
shap.summary_plot(
    spoofing_shap,
    X_test_combined.iloc[:500],
    plot_type="bar",
    show=False
)
plt.title("What drives Spoofing detection?")
plt.tight_layout()
plt.savefig("shap_spoofing.png", dpi=150)
#plt.show()


#isolation forest setup + evaluation
normal_mask = encoded_labels == 2
X_normal_only = scaled_features[normal_mask][network_columns + bio_columns]
X_test_combined = X_test_raw[network_columns + bio_columns]
isoforest = IsolationForest(contamination=0.125, random_state=67, n_jobs=-1)
isoforest.fit(X_normal_only)
iso_preds_raw = isoforest.predict(X_test_combined)
iso_binary = (iso_preds_raw == -1).astype(int)
y_test_binary = (y_test_raw != 2).astype(int)
#looks at metrics of isolation forest
print("\n Isolation Forest")
print(classification_report(y_test_binary, iso_binary, target_names=['normal', 'attack']))
iso_scores = isoforest.decision_function(X_test_combined)
print(f"Mean anomaly score for attacks: {iso_scores[y_test_binary==1].mean():.4f}")
print(f"Mean anomaly score for normal:  {iso_scores[y_test_binary==0].mean():.4f}")

#autoencoder setup (compress down to 8-dim bottleneck then reconstruct)
n_features = len(network_columns + bio_columns)
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
X_normal_np = X_normal_only.values
autoencoder.fit(
    X_normal_np, X_normal_np,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

#analyzing reconstruction error on test set
X_test_np = X_test_combined.values
X_reconstructed = autoencoder.predict(X_test_np)
recon_error = np.mean((X_test_np - X_reconstructed) ** 2, axis=1)

#threshold is 99th percentile of normal reconstruction error
normal_test_mask = y_test_binary == 0
threshold = np.percentile(recon_error[normal_test_mask], 99)
ae_binary = (recon_error > threshold).astype(int)

print(f"\nAutoencoder with threshold={threshold:.4f}")
print(classification_report(y_test_binary, ae_binary, target_names=['normal', 'attack']))
print(f"AUROC score: {roc_auc_score(y_test_binary, recon_error):.4f}")


# reconstruction error distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(recon_error[y_test_binary==0], bins=60, alpha=0.6, color='#036f2c', label='Normal')
axes[0].hist(recon_error[y_test_binary==1], bins=60, alpha=0.6, color='#d9534f', label='Attack')
axes[0].axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.3f}')
axes[0].set_xlabel("Reconstruction Error (MSE)")
axes[0].set_ylabel("Sample Count")
axes[0].set_title("Autoencoder: Normal vs Attack Reconstruction Error")
axes[0].legend()

#per-feature (bio vs network) reconstruction error 
per_feature_error = np.mean((X_test_np - X_reconstructed) ** 2, axis=0)
feature_names = network_columns + bio_columns
feat_err_series = pd.Series(per_feature_error, index=feature_names).nlargest(20).sort_values()
colors = ["#036f2c" if f in bio_columns else "#d78602" for f in feat_err_series.index]
feat_err_series.plot(kind='barh', color=colors, ax=axes[1])
axes[1].set_title("Per-Feature Reconstruction Error (Biometric=Green, Network=Orange)")
axes[1].set_xlabel("Mean Squared Error")
plt.tight_layout()
plt.savefig("autoencoder_results.png", dpi=150)
plt.show()

#shows random forest results vs iso forest vs autoencoder results
rf_binary_prob = 1 - model_comb.predict_proba(X_test_combined)[:, 2]
iso_score_normalized = -iso_scores

comparison = {'Model': ['Random Forest\n(Supervised)', 'Isolation Forest', 'Autoencoder'],
    'AUROC': [
        roc_auc_score(y_test_binary, rf_binary_prob),
        roc_auc_score(y_test_binary, iso_score_normalized),
        roc_auc_score(y_test_binary, recon_error)
    ],
    'Color': ['#2c5fd7', '#d78602', '#8e44ad']
}
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(comparison['Model'], comparison['AUROC'], color=comparison['Color'], width=0.5)
ax.set_ylim(0.5, 1.05)
ax.set_title("AUROC comparison of supervised vs unsupervised attack detection")
ax.set_ylabel("AUROC")
for bar, val in zip(bars, comparison['AUROC']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("model_comparison_all.png", dpi=150)
plt.show()


#severity scoring for normal vs spoofing vs data alteration using 3 signals (1 from RF attack probability, 2 is normalized isolation forest results, 3 is normalized AE reconstruction error)
X_full_combined = scaled_features[network_columns + bio_columns]
rf_attack_prob = 1 - model_comb.predict_proba(X_full_combined)[:, 2]

iso_full = isoforest.decision_function(X_full_combined)
iso_norm = 1 - (iso_full - iso_full.min()) / (iso_full.max() - iso_full.min())

ae_full = np.mean((X_full_combined.values - autoencoder.predict(X_full_combined.values)) ** 2, axis=1)
ae_norm = (ae_full - ae_full.min()) / (ae_full.max() - ae_full.min())

#severity is mostly weighted on random forest due to high AUROC scores and then slightly on isoforest and autoencoder results
severity = (0.5 * rf_attack_prob + 0.3 * iso_norm + 0.2 * ae_norm) * 100


df_severity = pd.DataFrame({
    'True Label': label_encoder.inverse_transform(encoded_labels),
    'Severity':     severity.round(1),
    'RF Signal':    (rf_attack_prob * 100).round(1),
    'IF Signal':    (iso_norm * 100).round(1),
    'AE Signal':    (ae_norm * 100).round(1)
})

df_severity['Alert Level'] = pd.cut(
    df_severity['Severity'],
    bins=[0, 30, 65, 100],
    labels=['Low', 'Medium', 'High']
)

print("\nSeverity Score by Class")
print(df_severity.groupby('True Label')['Severity'].describe().round(2))
print("\nAlert Level Distribution")
print(pd.crosstab(df_severity['True Label'], df_severity['Alert Level']))