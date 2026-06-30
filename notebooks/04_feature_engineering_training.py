import io
import json
import os
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import adlfs
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings('ignore')

# ── CONFIGURATION ──
STORAGE_ACCOUNT = "dlnycproject"
STORAGE_KEY = os.environ.get("AZURE_STORAGE_KEY", "<YOUR_AZURE_STORAGE_KEY>")

# Connect to ADLS Gen2
blob_service = BlobServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=STORAGE_KEY
)
container = blob_service.get_container_client("gold")

# Load and combine all parquet files from the gold layer
blobs = [b.name for b in container.list_blobs(name_starts_with="ml_features/") if b.name.endswith(".parquet")]
print(f"Found {len(blobs)} parquet files")

dfs = []
for blob_name in blobs:
    blob_client = container.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    dfs.append(pd.read_parquet(io.BytesIO(data)))

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# ── DATA CLEANING & FEATURE ENGINEERING ──
df_ml = df.copy()
df_ml = df_ml[~df_ml['borough'].isin(['UNSPECIFIED', None, ''])]
df_ml = df_ml.dropna(subset=['borough', 'year', 'week'])
df_ml = df_ml.fillna(0)

# Encode categorical boroughs to numerical values
borough_map = {
    'BRONX': 0, 'BROOKLYN': 1, 'MANHATTAN': 2,
    'QUEENS': 3, 'STATEN ISLAND': 4
}
df_ml['borough_encoded'] = df_ml['borough'].map(borough_map)
df_ml = df_ml.dropna(subset=['borough_encoded'])
df_ml['borough_encoded'] = df_ml['borough_encoded'].astype(int)

# Create next week's crimes to establish target (avoiding data leakage by shifting crimes)
df_ml = df_ml.sort_values(['borough', 'year', 'week']).reset_index(drop=True)
df_ml['next_week_crimes'] = df_ml.groupby('borough')['total_crimes'].shift(-1)
df_ml['target'] = (df_ml['next_week_crimes'] > 500).astype(int)
df_ml = df_ml.dropna(subset=['next_week_crimes'])

feature_cols = [
    'borough_encoded',
    'week',
    'month',
    'total_complaints',
    'unique_complaint_types',
    'complaint_crime_ratio',
]

X = df_ml[feature_cols]
y = df_ml['target']
groups = df_ml['borough_encoded']

# ── MODEL TRAINING & LEAVE-ONE-BOROUGH-OUT CROSS-VALIDATION ──
logo = LeaveOneGroupOut()
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss'
)

# Perform Leave-One-Borough-Out cross-validation
y_pred = cross_val_predict(model, X, y, groups=groups, cv=logo)

acc  = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec  = recall_score(y, y_pred, zero_division=0)
f1   = f1_score(y, y_pred, zero_division=0)

print("=== Cross Validation Metrics ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(classification_report(y, y_pred, target_names=['Normal Week', 'High Crime Week']))

# Fit final model on the full dataset
model.fit(X, y)
model.save_model("nyc_crime_model.json")

# ── FEATURE IMPORTANCE PLOT ──
fig, ax = plt.subplots(figsize=(10, 6))
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

importance.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Importance — NYC Crime Spike Prediction Model', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ── EXPORT TO ADLS MODEL STORE ──
fs = adlfs.AzureBlobFileSystem(
    account_name=STORAGE_ACCOUNT,
    account_key=STORAGE_KEY
)

# Upload the trained model
with fs.open("gold/models/nyc_crime_model.json", "wb") as f:
    with open("nyc_crime_model.json", "rb") as local_f:
        f.write(local_f.read())

# Write out current model metrics
metrics = {
    "accuracy":   round(acc, 4),
    "precision":  round(prec, 4),
    "recall":     round(rec, 4),
    "f1_score":   round(f1, 4),
    "model":      "XGBoostClassifier",
    "validation": "Leave-One-Borough-Out CV",
    "target":     "next_week_high_crime (total_crimes > 500)",
    "features":   feature_cols,
    "trained_on": str(pd.Timestamp.now())
}

with fs.open("gold/models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✓ Model & metrics exported to ADLS model store successfully.")
