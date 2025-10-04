import os, pandas as pd, joblib
from simple_salesforce import Salesforce
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# --- Salesforce connection
sf = Salesforce(
    username=os.getenv("SF_USERNAME"),
    password=os.getenv("SF_PASSWORD"),
    security_token=os.getenv("SF_SECURITY_TOKEN"),
    domain=os.getenv("SF_DOMAIN", "login")
)

# --- Pull Telemetry
telemetry = sf.query_all("""
SELECT Id, Asset__c, Ingested_At__c,
       Temperature__c, Vibration_Level__c, Voltage__c, Operating_Hours__c
FROM Telemetry__c
WHERE Asset__c != NULL
""")["records"]
t_df = pd.DataFrame(telemetry)
if t_df.empty:
    raise SystemExit("No Telemetry__c data found.")
t_df["Ingested_At__c"] = pd.to_datetime(t_df["Ingested_At__c"])

# --- Pull Work Orders (proxy: maintenance date ~ LastModifiedDate)
work_orders = sf.query_all("""
SELECT Id, AssetId, Status, CreatedDate, LastModifiedDate
FROM WorkOrder
WHERE AssetId != NULL
""")["records"]
wo_df = pd.DataFrame(work_orders)
if wo_df.empty:
    raise SystemExit("No WorkOrder data found.")
wo_df["LastModifiedDate"] = pd.to_datetime(wo_df["LastModifiedDate"])

# --- Build label: days to next WO after telemetry
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

t_df = t_df.dropna(subset=["label_days"]).copy()
if t_df.empty:
    raise SystemExit("No labeled rows; create/ensure WOs occur after telemetry timestamps.")

# --- Features
features = ["Temperature__c", "Vibration_Level__c", "Voltage__c", "Operating_Hours__c"]
for c in features:
    t_df[c] = pd.to_numeric(t_df[c], errors="coerce").fillna(0.0)

X, y = t_df[features].values, t_df["label_days"].values

# --- Train & persist
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=13)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=13))
])
pipe.fit(Xtr, ytr)

os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/regressor.pkl")
print("Saved models/regressor.pkl")
