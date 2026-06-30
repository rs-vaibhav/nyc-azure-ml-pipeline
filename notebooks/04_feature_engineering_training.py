import io
import json
import os
import warnings

import adlfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── CONFIGURATION ──
STORAGE_ACCOUNT = "dlnycproject"
STORAGE_KEY = os.environ.get("AZURE_STORAGE_KEY", "<YOUR_AZURE_STORAGE_KEY>")

BOROUGH_MAP = {
    "BRONX": 0, "BROOKLYN": 1, "MANHATTAN": 2,
    "QUEENS": 3, "STATEN ISLAND": 4,
}

FEATURE_COLS = [
    "borough_encoded",
    "week",
    "month",
    "total_complaints",
    "unique_complaint_types",
    "complaint_crime_ratio",
]

# ── DATA LOADING ──
blob_service = BlobServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=STORAGE_KEY,
)
container = blob_service.get_container_client("gold")

blobs = [
    b.name
    for b in container.list_blobs(name_starts_with="ml_features/")
    if b.name.endswith(".parquet")
]
print(f"Found {len(blobs)} parquet files")

dfs = []
for blob_name in blobs:
    data = container.get_blob_client(blob_name).download_blob().readall()
    dfs.append(pd.read_parquet(io.BytesIO(data)))

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# ── DATA CLEANING & FEATURE ENGINEERING ──
df_ml = df.copy()
df_ml = df_ml[~df_ml["borough"].isin(["UNSPECIFIED", None, ""])]
df_ml = df_ml.dropna(subset=["borough", "year", "week"]).fillna(0)

df_ml["borough_encoded"] = df_ml["borough"].map(BOROUGH_MAP)
df_ml = df_ml.dropna(subset=["borough_encoded"])
df_ml["borough_encoded"] = df_ml["borough_encoded"].astype(int)

df_ml = df_ml.sort_values(["borough", "year", "week"]).reset_index(drop=True)
df_ml["next_week_crimes"] = df_ml.groupby("borough")["total_crimes"].shift(-1)
df_ml["target"] = (df_ml["next_week_crimes"] > 500).astype(int)
df_ml = df_ml.dropna(subset=["next_week_crimes"])

X = df_ml[FEATURE_COLS]
y = df_ml["target"]
groups = df_ml["borough_encoded"]

# ── MODEL TRAINING (Leave-One-Borough-Out Cross-Validation) ──
logo = LeaveOneGroupOut()
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    eval_metric="logloss",
)

y_pred = cross_val_predict(model, X, y, groups=groups, cv=logo)

acc  = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec  = recall_score(y, y_pred, zero_division=0)
f1   = f1_score(y, y_pred, zero_division=0)

print("=== Cross-Validation Metrics ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(classification_report(y, y_pred, target_names=["Normal Week", "High Crime Week"]))

# Fit final model on all available data
model.fit(X, y)
model.save_model("nyc_crime_model.json")

# ── FEATURE IMPORTANCE PLOT ──
fig, ax = plt.subplots(figsize=(10, 6))
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
importance.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Feature Importance — NYC Crime Spike Prediction", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.axvline(x=0.1, color="red", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ── EXPORT MODEL & METRICS TO ADLS ──
fs = adlfs.AzureBlobFileSystem(account_name=STORAGE_ACCOUNT, account_key=STORAGE_KEY)

with fs.open("gold/models/nyc_crime_model.json", "wb") as remote_f:
    with open("nyc_crime_model.json", "rb") as local_f:
        remote_f.write(local_f.read())

metrics = {
    "accuracy":   round(acc, 4),
    "precision":  round(prec, 4),
    "recall":     round(rec, 4),
    "f1_score":   round(f1, 4),
    "model":      "XGBoostClassifier",
    "validation": "Leave-One-Borough-Out CV",
    "target":     "next_week_high_crime (total_crimes > 500)",
    "features":   FEATURE_COLS,
    "trained_on": str(pd.Timestamp.now()),
}

with fs.open("gold/models/metrics.json", "w") as remote_f:
    json.dump(metrics, remote_f, indent=2)

print("✓ Model & metrics exported to ADLS model store successfully.")
