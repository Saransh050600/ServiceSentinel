import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib
from simple_salesforce import Salesforce

# --- 1. Connect to Salesforce ---
sf = Salesforce(
    username='saransh050600@curious-fox-h8zs8p.com',  # Replace with your Salesforce username
    password='Netsettere@5',                         # Replace with your Salesforce password
    security_token='XtQCNJRs6aIPiwPDAKJeR1Jb',        # Replace with your Salesforce security token
    domain='login'                                   # Use 'test' for sandbox orgs
)

# --- 2. Query Telemetry Data ---
query = """
SELECT Asset__r.AssetType__c, Current__c, Flow_Rate__c, Frequency__c,
       Health_Score__c, Load__c, Oil_Level__c, Operating_Hours__c,
       Pressure__c, RPM__c, Remaining_Useful_Life_Hrs__c, Risk_Level__c,
       Steam_Flow__c, Temperature__c, Vibration_Level__c, Voltage__c
FROM Telemetry__c
"""
records = sf.query_all(query)['records']

# --- 3. Convert to DataFrame and Clean ---
df = pd.DataFrame(records).drop(columns=['attributes'])
df.fillna(0, inplace=True)

# --- 4. Flatten AssetType ---
if 'Asset__r' in df.columns:
    df['AssetType'] = df['Asset__r'].apply(
        lambda x: x.get('AssetType__c') if isinstance(x, dict) else None
    )
elif 'Asset__r.AssetType__c' in df.columns:
    df['AssetType'] = df['Asset__r.AssetType__c']
else:
    raise KeyError("Asset type field not found.")

df['AssetType'] = df['AssetType'].fillna('Unknown').astype(str).str.strip()
df['Risk_Level__c'] = df['Risk_Level__c'].astype(str).str.strip()

# --- 5. Encode Labels ---
asset_encoder = LabelEncoder()
df['AssetType_enc'] = asset_encoder.fit_transform(df['AssetType'])

risk_encoder = LabelEncoder()
df['Risk_enc'] = risk_encoder.fit_transform(df['Risk_Level__c'])

# --- 6. Features and Targets ---
exclude_cols = ['Risk_Level__c', 'Risk_enc', 'Remaining_Useful_Life_Hrs__c',
                'Asset__r', 'AssetType']
numeric_cols = [
    col for col in df.columns
    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
]
if 'AssetType_enc' not in numeric_cols:
    numeric_cols.insert(0, 'AssetType_enc')

print(f"Features used: {numeric_cols}")

X = df[numeric_cols]
y_class = df['Risk_enc']  # Classification target
y_rul = df['Remaining_Useful_Life_Hrs__c']  # Regression target

# --- 7. Split Data ---
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_class, test_size=0.2, stratify=df['AssetType'], random_state=42
)

# Align y_rul with the same split indices
y_rul_train = y_rul.loc[X_train.index]
y_rul_test = y_rul.loc[X_test.index]

# --- 8. Train Models ---
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_class_train)

rul_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rul_reg.fit(X_train, y_rul_train)

# --- 9. Evaluate ---
y_class_pred = clf.predict(X_test)
print("\n📊 Classification Report (Risk Level):")
print(classification_report(y_class_test, y_class_pred, target_names=risk_encoder.classes_))

y_rul_pred = rul_reg.predict(X_test)
mae = mean_absolute_error(y_rul_test, y_rul_pred)
r2 = r2_score(y_rul_test, y_rul_pred)
print(f"\n📈 RUL Regression - MAE: {mae:.2f}, R²: {r2:.2f}")

# --- 10. Save Models ---
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/asset_risk_model.pkl')
joblib.dump(rul_reg, 'models/rul_model.pkl')
joblib.dump(asset_encoder, 'models/asset_encoder.pkl')
joblib.dump(risk_encoder, 'models/risk_encoder.pkl')

print("\n✅ Models and encoders saved in /models/")
