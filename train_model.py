import os
import json
import pandas as pd
import numpy as np
import joblib
from simple_salesforce import Salesforce
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()

SF_USERNAME = os.getenv("SF_USERNAME")
SF_PASSWORD = os.getenv("SF_PASSWORD")
SF_SECURITY_TOKEN = os.getenv("SF_SECURITY_TOKEN")
SF_DOMAIN = os.getenv("SF_DOMAIN", "login")

MIN_SAMPLES_PER_TYPE = int(os.getenv("MIN_SAMPLES_PER_TYPE", 10))
MODEL_DIR = "models"
COMPLETION_STATUSES = {"Completed", "Closed"}
WO_EVENT_FIELD = os.getenv("WO_EVENT_FIELD", "LastModifiedDate")  # or CreatedDate / your custom field

if not (SF_USERNAME and SF_PASSWORD and SF_SECURITY_TOKEN):
    raise SystemExit("Missing Salesforce credentials. Set SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN.")

sf = Salesforce(username=SF_USERNAME, password=SF_PASSWORD, security_token=SF_SECURITY_TOKEN, domain=SF_DOMAIN)

telemetry = sf.query_all("""
SELECT Id, Asset__c, Ingested_At__c,
       Temperature__c, Vibration_Level__c, Voltage__c, Operating_Hours__c,
       Asset__r.AssetType__c
FROM Telemetry__c
WHERE Asset__c != NULL
""")["records"]

t_df = pd.DataFrame(telemetry)
if t_df.empty:
    raise SystemExit("No Telemetry__c data found.")

t_df["Ingested_At__c"] = pd.to_datetime(t_df["Ingested_At__c"], errors="coerce").dt.tz_localize(None)

def extract_asset_type(val):
    if isinstance(val, dict):
        return val.get("AssetType__c")
    return None

t_df["AssetType"] = t_df.get("Asset__r", pd.Series([None] * len(t_df))).apply(extract_asset_type)

work_orders = sf.query_all("""
SELECT Id, AssetId, Status, CreatedDate, LastModifiedDate
FROM WorkOrder
WHERE AssetId != NULL
  AND (Status = 'Completed' OR Status = 'Closed')
""")["records"]

wo_df = pd.DataFrame(work_orders)
if wo_df.empty:
    raise SystemExit("No Completed or Closed WorkOrder data found.")

for col in ("CreatedDate", "LastModifiedDate"):
    if col in wo_df.columns:
        wo_df[col] = pd.to_datetime(wo_df[col], errors="coerce").dt.tz_localize(None)
    else:
        wo_df[col] = pd.NaT

if WO_EVENT_FIELD not in wo_df.columns:
    print(f"[WARN] WO_EVENT_FIELD '{WO_EVENT_FIELD}' not in results; falling back to LastModifiedDate/CreatedDate.")
    WO_EVENT_FIELD = "LastModifiedDate"

wo_df["EventTime"] = wo_df[WO_EVENT_FIELD].where(wo_df[WO_EVENT_FIELD].notna(), wo_df["CreatedDate"])
wo_df = wo_df[wo_df["EventTime"].notna()].copy()

def next_maintenance_days(asset_id, ing_time):
    if pd.isna(ing_time) or asset_id is None:
        return np.nan
    rows = wo_df[(wo_df["AssetId"] == asset_id) & (wo_df["EventTime"] > ing_time)]
    if rows.empty:
        return np.nan
    next_time = rows["EventTime"].min()
    delta_days = (next_time - ing_time).days
    return float(max(delta_days, 0))

t_df["label_days"] = t_df.apply(lambda r: next_maintenance_days(r["Asset__c"], r["Ingested_At__c"]), axis=1)
t_df = t_df.dropna(subset=["label_days", "AssetType"]).copy()
if t_df.empty:
    raise SystemExit("No labeled telemetry rows with valid AssetType found (no later completed/closed WO).")

FEATURES = ["Temperature__c", "Vibration_Level__c", "Voltage__c", "Operating_Hours__c"]
for c in FEATURES:
    t_df[c] = pd.to_numeric(t_df[c], errors="coerce").fillna(0.0)

os.makedirs(MODEL_DIR, exist_ok=True)

asset_types = sorted(t_df["AssetType"].dropna().unique())
print(f"Found {len(asset_types)} Asset types: {asset_types}")

for asset_type, grp in t_df.groupby("AssetType"):
    n = len(grp)
    if n < MIN_SAMPLES_PER_TYPE:
        print(f"Skipping '{asset_type}': insufficient samples ({n} < {MIN_SAMPLES_PER_TYPE})")
        continue

    X = grp[FEATURES].values
    y = grp["label_days"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=13)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=13))
    ])
    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xte)
    mae = float(mean_absolute_error(yte, ypred))
    r2 = float(r2_score(yte, ypred))

    safe_type = str(asset_type).replace(" ", "_")
    model_path = os.path.join(MODEL_DIR, f"regressor_{safe_type}.pkl")
    joblib.dump(pipe, model_path)

    meta = {
        "assetType": asset_type,
        "features": ["Temperature", "Vibration_Level", "Voltage", "Operating_Hours"],
        "samples_total": int(n),
        "samples_train": int(len(Xtr)),
        "samples_test": int(len(Xte)),
        "metrics": {"MAE_days": mae, "R2": r2},
        "wo_event_field": WO_EVENT_FIELD
    }
    with open(os.path.join(MODEL_DIR, f"regressor_{safe_type}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Trained '{asset_type}' → {model_path} | MAE(days)={mae:.3f}, R²={r2:.3f} (event={WO_EVENT_FIELD})")

print("🎯 Training complete for all available Asset types.")
