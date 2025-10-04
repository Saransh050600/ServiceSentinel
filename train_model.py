import os
import pandas as pd
from datetime import datetime, timedelta
from simple_salesforce import Salesforce
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from dotenv import load_dotenv

load_dotenv()

SF_USERNAME = os.getenv("SF_USERNAME")
SF_PASSWORD = os.getenv("SF_PASSWORD")
SF_SECURITY_TOKEN = os.getenv("SF_SECURITY_TOKEN")
SF_DOMAIN = os.getenv("SF_DOMAIN", "login")

sf = Salesforce(username=SF_USERNAME, password=SF_PASSWORD,
                security_token=SF_SECURITY_TOKEN, domain=SF_DOMAIN)

# ---- 1) Pull telemetry
# Keep columns you actually created in Salesforce
telemetry = sf.query_all("""
SELECT Id, Asset__c, Ingested_At__c,
       Temperature__c, Vibration_Level__c, Voltage__c, Operating_Hours__c
FROM Telemetry__c
WHERE Asset__c != NULL
""")["records"]
t_df = pd.DataFrame(telemetry)
if t_df.empty:
    raise SystemExit("No Telemetry__c data found.")

# Normalize SF date fields
t_df["Ingested_At__c"] = pd.to_datetime(t_df["Ingested_At__c"])

# ---- 2) Pull Work Orders (assuming ClosedDate or CompletedDate is populated)
# Adjust WHERE as needed for your org (Status='Closed' etc.)
work_orders = sf.query_all("""
SELECT Id, AssetId, Subject, Status, CreatedDate, LastModifiedDate
FROM WorkOrder
WHERE AssetId != NULL
""")["records"]
wo_df = pd.DataFrame(work_orders)
if wo_df.empty:
    raise SystemExit("No WorkOrder data found. You need some maintenance history to train.")

# Derive a "closed date" proxy (if you have a real field, use it)
# We'll use LastModifiedDate as a stand-in for 'maintenance performed date' when status indicates completion.
wo_df["LastModifiedDate"] = pd.to_datetime(wo_df["LastModifiedDate"])
wo_df["CreatedDate"] = pd.to_datetime(wo_df["CreatedDate"])

# ---- 3) Build label: next maintenance date after telemetry -> days_to_next_maintenance
def next_maintenance_days(asset_id, ingested_at):
    # Future WOs for this asset
    rows = wo_df[(wo_df["AssetId"] == asset_id) & (wo_df["LastModifiedDate"] > ingested_at)]
    if rows.empty:
        return None
    next_date = rows["LastModifiedDate"].min()
    delta_days = (next_date - ingested_at).days
    return max(delta_days, 0)

t_df["label_days"] = t_df.apply(
    lambda r: next_maintenance_days(r["Asset__c"], r["Ingested_At__c"]), axis=1
)

# Drop rows without labels
t_df = t_df.dropna(subset=["label_days"]).copy()
if t_df.empty:
    raise SystemExit("No labeled telemetry rows (couldn't find future WOs).")

# ---- 4) Feature engineering
features = ["Temperature__c", "Vibration_Level__c", "Voltage__c", "Operating_Hours__c"]
for col in features:
    t_df[col] = pd.to_numeric(t_df[col], errors="coerce").fillna(0.0)

X = t_df[features].values
y = t_df["label_days"].astype(float).values

# ---- 5) Train/validate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=13,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
mae = abs(pred - y_test).mean()
print(f"MAE (days): {mae:.2f}")

# ---- 6) Persist model
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/regressor.pkl")
print("Saved models/regressor.pkl")
