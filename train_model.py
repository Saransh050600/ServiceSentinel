import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser(description="Train HGB regressors per AssetType from Excel (no Salesforce).")
    ap.add_argument("--excel", required=True, help="Path to the Excel file containing both sheets")
    ap.add_argument("--telemetry-sheet", default="Telemetry", help="Telemetry sheet name")
    ap.add_argument("--workorders-sheet", default="WorkOrders", help="WorkOrders sheet name")
    ap.add_argument("--wo-event-field", default="LastModifiedDate",
                    help="WO event time column to use (LastModifiedDate or CreatedDate)")
    ap.add_argument("--min-samples", type=int, default=10, help="Minimum labeled rows per AssetType to fit a model")
    ap.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    ap.add_argument("--random-state", type=int, default=13, help="Random seed")
    ap.add_argument("--model-dir", default="models", help="Directory to write models and meta")
    return ap.parse_args()

COMPLETION_STATUSES = {"Completed", "Closed"}

FEATURES = ["Temperature__c", "Vibration_Level__c", "Voltage__c", "Operating_Hours__c"]

def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)

def next_maintenance_days_factory(wo_by_asset: dict, event_col: str):
    def _fn(asset_id, ing_time):
        if pd.isna(ing_time) or pd.isna(asset_id):
            return np.nan
        g = wo_by_asset.get(asset_id)
        if g is None or g.empty:
            return np.nan
        # find first WO strictly after the telemetry timestamp
        idx = g[event_col].searchsorted(ing_time, side="right")
        if idx >= len(g):
            return np.nan
        next_time = g[event_col].iloc[idx]
        return float(max((next_time - ing_time).days, 0))
    return _fn

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # ---- Read Excel ----
    try:
        t_df = pd.read_excel(args.excel, sheet_name=args.telemetry_sheet)
    except Exception as e:
        raise SystemExit(f"Failed reading telemetry sheet '{args.telemetry_sheet}': {e}")

    try:
        wo_df = pd.read_excel(args.excel, sheet_name=args.workorders_sheet)
    except Exception as e:
        raise SystemExit(f"Failed reading workorders sheet '{args.workorders_sheet}': {e}")

    if t_df.empty:
        raise SystemExit("No telemetry rows found.")
    if wo_df.empty:
        raise SystemExit("No work order rows found.")

    # ---- Basic column checks (exact names as requested) ----
    tel_required = ["Asset__c", "AssetType__c", "Ingested_At__c"] + FEATURES
    missing_tel = [c for c in tel_required if c not in t_df.columns]
    if missing_tel:
        raise SystemExit(f"Telemetry sheet is missing columns: {missing_tel}")

    wo_required = ["Id", "AssetId", "Status", "CreatedDate", "LastModifiedDate"]
    missing_wo = [c for c in wo_required if c not in wo_df.columns]
    if missing_wo:
        raise SystemExit(f"WorkOrders sheet is missing columns: {missing_wo}")

    # ---- Normalize dtypes ----
    t_df["Ingested_At__c"] = ensure_datetime(t_df["Ingested_At__c"])
    for c in FEATURES:
        t_df[c] = pd.to_numeric(t_df[c], errors="coerce").fillna(0.0)

    # event time for WO
    wo_df["CreatedDate"] = ensure_datetime(wo_df["CreatedDate"])
    wo_df["LastModifiedDate"] = ensure_datetime(wo_df["LastModifiedDate"])

    if args.wo_event_field not in wo_df.columns:
        print(f"[WARN] WO event field '{args.wo_event_field}' not found. Falling back to LastModifiedDate/CreatedDate.")
        wo_event_col = "LastModifiedDate"
    else:
        wo_event_col = args.wo_event_field

    wo_df["EventTime"] = wo_df[wo_event_col].where(wo_df[wo_event_col].notna(), wo_df["CreatedDate"])
    wo_df = wo_df[(wo_df["AssetId"].notna()) & (wo_df["EventTime"].notna())].copy()

    # Keep Completed/Closed WOs only
    wo_df = wo_df[wo_df["Status"].astype(str).isin(COMPLETION_STATUSES)].copy()
    if wo_df.empty:
        raise SystemExit("No Completed/Closed WorkOrders with valid timestamps.")

    # ---- Pre-index WO by AssetId ----
    wo_df = wo_df.sort_values("EventTime")
    wo_by_asset = dict(tuple(wo_df.groupby("AssetId")))
    get_label = next_maintenance_days_factory(wo_by_asset, "EventTime")

    # ---- Build labels (days to next maintenance) ----
    t_df["label_days"] = t_df.apply(lambda r: get_label(r["Asset__c"], r["Ingested_At__c"]), axis=1)
    t_df = t_df.dropna(subset=["label_days", "AssetType__c"]).copy()
    if t_df.empty:
        raise SystemExit("No labeled telemetry rows with valid AssetType__c found (no later Completed/Closed WO).")

    asset_types = sorted(t_df["AssetType__c"].dropna().astype(str).unique())
    print(f"Found {len(asset_types)} Asset types: {asset_types}")

    # ---- Train per AssetType ----
    for asset_type, grp in t_df.groupby("AssetType__c"):
        grp = grp.dropna(subset=FEATURES + ["label_days"])
        n = len(grp)
        if n < args.min_samples:
            print(f"Skipping '{asset_type}': insufficient samples ({n} < {args.min_samples})")
            continue

        X = grp[FEATURES].values
        y = grp["label_days"].values

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

        # HGB does not require feature scaling
        model = HistGradientBoostingRegressor(random_state=args.random_state)
        model.fit(Xtr, ytr)

        ypred = model.predict(Xte)
        mae = float(mean_absolute_error(yte, ypred))
        r2 = float(r2_score(yte, ypred))

        safe_type = str(asset_type).replace(" ", "_")
        model_path = os.path.join(args.model_dir, f"regressor_{safe_type}.pkl")
        joblib.dump(model, model_path)

        meta = {
            "assetType": asset_type,
            "features": ["Temperature", "Vibration_Level", "Voltage", "Operating_Hours"],
            "samples_total": int(n),
            "samples_train": int(len(Xtr)),
            "samples_test": int(len(Xte)),
            "metrics": {"MAE_days": mae, "R2": r2},
            "wo_event_field": wo_event_col,
            "model_type": "HistGradientBoostingRegressor"
        }
        with open(os.path.join(args.model_dir, f"regressor_{safe_type}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Trained '{asset_type}' → {model_path} | MAE(days)={mae:.3f}, R²={r2:.3f} (event={wo_event_col})")

    print("🎯 Training complete for all available Asset types.")

if __name__ == "__main__":
    main()