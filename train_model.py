import os
import pandas as pd
import joblib
from simple_salesforce import Salesforce
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# -----------------------------
# Load Salesforce credentials
# -----------------------------
load_dotenv()

sf = Salesforce(
    username=os.getenv("SF_USERNAME"),
    password=os.getenv("SF_PASSWORD"),
    security_token=os.getenv("SF_SECURITY_TOKEN"),
    domain=os.getenv("SF_DOMAIN", "login")
)

# -----------------------------
# Pull Telemetry with Asset Type
# -----------------------------
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

# Convert ingestion timestamp to datetime
t_df["Ingested_At__c"] = pd.to_datetime(t_df["Ingested_At__c"])

# Flatten Asset type from nested dict into simple column
def extract_asset_type(val):
    if isinstance(val, dict):
        return val.get("AssetType__c")
    return None

if "Asset__r" in t_df.columns:
    t_df["AssetType"] = t_df["Asset__r"].apply(extract_asset_type)
else:
    t_df["AssetType"] = None

# -----------------------------
# Pull Completed/Closed WorkOrders
# -----------------------------
work_orders = sf.query_all("""
SELECT Id, AssetId, Status, CreatedDate, LastModifiedDate
FROM WorkOrder
WHERE AssetId != NULL AND (Status = 'Completed' OR Status = 'Closed')
""")["records"]

wo_df = pd.DataFrame(work_orders)
if wo_df.empty:
    raise SystemExit("No Completed or Closed WorkOrder data found.")

wo_df["LastModifiedDate"] = pd.to_datetime(wo_df["LastModifiedDate"])

# -----------------------------
# Label: Days until next Completed WO after telemetry
# -----------------------------
def next_maintenance_days(asset_id, ing_at):
    rows = wo_df[(wo_df["AssetId"] == asset_id) & (wo_df["LastModifiedDate"] > ing_at)]
    if rows.empty:
        return None
    next_dt = rows["LastModifiedDate"].min()
    return max((next_dt - ing_at).days, 0)

t_df["label_days"] = t_df.apply(
    lambda r: next_maintenance_days(r["Asset__c"], r["Ingested_At__c"]),
    axis=1
).astype("float")

# Keep only labeled rows with valid AssetType
t_df = t_df.dropna(subset=["label_days", "AssetType"]).copy()
if t_df.empty:
    raise SystemExit("No labeled telemetry rows with valid AssetType found.")

# -----------------------------
# Feature setup
# -----------------------------
features = ["Temperature__c", "Vibration_Level__c", "Voltage__c", "Operating_Hours__c"]
for c in features:
    t_df[c] = pd.to_numeric(t_df[c], errors="coerce").fillna(0.0)

os.makedirs("models", exist_ok=True)

# -----------------------------
# Train separate model per AssetType
# -----------------------------
MIN_SAMPLES_PER_TYPE = 0  # only train if ≥10 samples per type

asset_types = sorted(t_df["AssetType"].unique())
print(f"Found {len(asset_types)} Asset types: {asset_types}")

for asset_type, df_group in t_df.groupby("AssetType"):
    n = len(df_group)
    if n < MIN_SAMPLES_PER_TYPE:
        print(f"Skipping '{asset_type}': insufficient samples ({n})")
        continue

    X, y = df_group[features].values, df_group["label_days"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=13)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=13))
    ])
    pipe.fit(Xtr, ytr)

    safe_type = str(asset_type).replace(" ", "_")
    model_path = f"models/regressor_{safe_type}.pkl"
    joblib.dump(pipe, model_path)

    print(f"✅ Trained model for '{asset_type}' → {model_path}")

print("🎯 Training complete for all available Asset types.")
