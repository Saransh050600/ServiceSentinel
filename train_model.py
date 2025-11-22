#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate 3 regressors (rf, hgb, xgb) per AssetType on Excel Telemetry + WorkOrders.
Pick best by Accuracy% (= 100 - MAPE%). Winner models are saved per AssetType.

Dynamic Features:
- All telemetry columns EXCEPT key columns + excluded label columns
- Per AssetType, only columns that have at least one non-NaN value used
"""

import os
import argparse, json, math
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor


# ------------------- CLI -------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate RF / HGB / XGB with dynamic telemetry features.")
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--workorders", required=True)
    ap.add_argument("--telemetry-sheet", default="Telemetry")
    ap.add_argument("--workorders-sheet", default="WorkOrders")
    ap.add_argument("--min-samples", type=int, default=10)
    # Back to 0.2 like your good run
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=13)
    ap.add_argument("--summary-path", default="excel_model_eval_summary.json")
    ap.add_argument("--no-summary", action="store_true")

    # where to save winner models / meta
    ap.add_argument("--models", default="models_eval",
                    help="Directory to save winner models per AssetType")
    ap.add_argument("--no-save-models", action="store_true",
                    help="If set, do not persist trained winner models.")
    return ap.parse_args()


# ----------------- Helpers ------------------
def ensure_datetime(series):
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def accuracy_from_mape(mp):
    if mp != mp:  # NaN
        return np.nan
    return max(0.0, 100.0 - float(mp))


# -------------------- MAIN -------------------
def main():
    args = parse_args()

    if not args.no_save_models:
        os.makedirs(args.models, exist_ok=True)

    # Load Excel
    t = pd.read_excel(args.telemetry, sheet_name=args.telemetry_sheet)
    wo = pd.read_excel(args.workorders, sheet_name=args.workorders_sheet)

    # Required Telemetry Columns
    TEL_KEY_COLS = ["Asset__c", "AssetType__c", "Ingested_At__c"]

    # Columns NEVER allowed as features (avoid leakage)
    EXCLUDE_COLS = ["label_days"]

    for c in TEL_KEY_COLS:
        if c not in t.columns:
            raise SystemExit(f"Missing required telemetry column: {c}")

    # Required WorkOrder Columns
    WO_KEY_COLS = ["AssetId", "Status", "CreatedDate"]
    for c in WO_KEY_COLS:
        if c not in wo.columns:
            raise SystemExit(f"Missing required WorkOrder column: {c}")

    # Parse timestamps
    t["Ingested_At__c"] = ensure_datetime(t["Ingested_At__c"])
    wo["CreatedDate"] = ensure_datetime(wo["CreatedDate"])

    # Potential features = all columns except keys + exclude list
    feature_candidates_all = [
        c for c in t.columns if c not in TEL_KEY_COLS + EXCLUDE_COLS
    ]

    # Convert telemetry feature columns to numeric
    for col in feature_candidates_all:
        t[col] = pd.to_numeric(t[col], errors="coerce")

    # Filter Completed/Closed WorkOrders
    wo["__status_norm"] = wo["Status"].astype(str).str.strip().str.lower()
    wo = wo[(wo["__status_norm"].isin(["completed", "closed"])) & wo["CreatedDate"].notna()]
    if wo.empty:
        raise SystemExit("No valid Completed/Closed WorkOrders found.")

    # Group WOs
    wo = wo.sort_values("CreatedDate")
    wo_by_asset = dict(tuple(wo.groupby("AssetId")))

    # Label function
    def next_wo_days(asset_id, ing_time):
        if pd.isna(asset_id) or pd.isna(ing_time):
            return np.nan
        g = wo_by_asset.get(asset_id)
        if g is None or g.empty:
            return np.nan
        idx = g["CreatedDate"].searchsorted(ing_time, side="right")
        if idx >= len(g):
            return np.nan
        next_time = g["CreatedDate"].iloc[idx]
        return float(max((next_time - ing_time).days, 0))

    # Build target variable
    t["label_days"] = t.apply(lambda r: next_wo_days(r["Asset__c"], r["Ingested_At__c"]), axis=1)
    t = t.dropna(subset=["label_days"])

    asset_types = sorted(t["AssetType__c"].dropna().astype(str).unique())
    print("Detected Asset Types:", asset_types)

    results = []

    # For global winner: sample-weighted accuracy per model
    global_model_stats = {
        "rf": {"sum_weighted_acc": 0.0, "sum_samples": 0},
        "xgb": {"sum_weighted_acc": 0.0, "sum_samples": 0},
        "hgb": {"sum_weighted_acc": 0.0, "sum_samples": 0},
    }

    for asset_type, grp in t.groupby("AssetType__c"):
        grp = grp.copy()

        # Select only columns where this asset type has non-zero/non-nan data
        feature_cols = [
            c for c in feature_candidates_all if grp[c].notna().any()
        ]

        # Safety: do not allow "label_days" as a column
        feature_cols = [c for c in feature_cols if c not in EXCLUDE_COLS]

        if not feature_cols:
            print(f"Skipping {asset_type}: no usable features.")
            continue

        grp = grp.dropna(subset=feature_cols + ["label_days"])
        n = len(grp)
        if n < args.min_samples:
            print(f"Skipping {asset_type}: insufficient samples ({n}).")
            continue

        print(f"\n=== AssetType: {asset_type} ===")
        print(f"Samples = {n}")
        print(f"Using Features: {feature_cols}")

        # Split
        X = grp[feature_cols].values
        y = grp["label_days"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

        # -------- Models (back to your original settings) --------
        models = {
            "rf": Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestRegressor(
                    n_estimators=500,
                    random_state=args.random_state
                ))
            ]),
            "hgb": Pipeline([
                ("hgb", HistGradientBoostingRegressor(
                    random_state=args.random_state
                ))
            ]),
            "xgb": Pipeline([
                ("scaler", StandardScaler()),
                ("xgb", XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=args.random_state,
                    tree_method="hist"
                ))
            ]),
        }

        metrics = {}

        for name, model in models.items():
            model.fit(Xtr, ytr)
            y_pred = model.predict(Xte)

            mae = mean_absolute_error(yte, y_pred)
            rmse = math.sqrt(mean_squared_error(yte, y_pred))
            r2 = r2_score(yte, y_pred)
            mp = mape(yte, y_pred)
            acc = accuracy_from_mape(mp)

            metrics[name] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
                "MAPE%": float(mp),
                "Accuracy%": float(acc)
            }

            # Update global stats (sample-weighted Accuracy)
            if not np.isnan(acc):
                global_model_stats[name]["sum_weighted_acc"] += acc * n
                global_model_stats[name]["sum_samples"] += n

        # Winner per asset type (pure max Accuracy)
        best_name = max(metrics.keys(), key=lambda k: metrics[k]["Accuracy%"])
        best = metrics[best_name]

        print("Model    MAE       RMSE      R2        MAPE%     Accuracy%")
        for name in ["rf", "xgb", "hgb"]:
            m = metrics[name]
            print(f"{name:<7} {m['MAE']:<9.4f} {m['RMSE']:<9.4f} {m['R2']:<9.4f} {m['MAPE%']:<9.2f} {m['Accuracy%']:<9.2f}")
        print(f"--> Winner: {best_name} (Accuracy={best['Accuracy%']:.2f}%)")

        model_path = None
        meta_path = None

        # -------- Save winner model + meta (per AssetType) --------
        if not args.no_save_models:
            winner_pipeline = models[best_name]

            safe_type = str(asset_type).replace(" ", "_")
            model_filename = f"regressor_{safe_type}.pkl"
            model_path = os.path.join(args.models, model_filename)
            joblib.dump(winner_pipeline, model_path)

            # Determine underlying estimator type for meta
            est = None
            if best_name in winner_pipeline.named_steps:
                est = winner_pipeline.named_steps[best_name]
            else:
                est = list(winner_pipeline.named_steps.values())[-1]
            model_type = type(est).__name__ if est is not None else "Unknown"

            meta = {
                "assetType": asset_type,
                "features": feature_cols,
                "samples_total": int(n),
                "samples_train": int(len(Xtr)),
                "samples_test": int(len(Xte)),
                "metrics": best,  # already has MAE, RMSE, R2, MAPE%, Accuracy%
                "model_type": model_type,
                "winner_key": best_name
            }

            meta_filename = f"regressor_{safe_type}.meta.json"
            meta_path = os.path.join(args.models, meta_filename)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"💾 Saved winner for '{asset_type}' → {model_path}")
            print(f"📝 Meta → {meta_path}")

        results.append({
            "assetType": asset_type,
            "samples": n,
            "features": feature_cols,
            "metrics": metrics,
            "winner": {
                "name": best_name,
                "metrics": best,
                "model_path": model_path,
                "meta_path": meta_path
            }
        })

    # ---------- Global combined winner ----------
    combined = {}
    for name, stats in global_model_stats.items():
        if stats["sum_samples"] > 0:
            avg_acc = stats["sum_weighted_acc"] / stats["sum_samples"]
        else:
            avg_acc = float("nan")
        combined[name] = {
            "weighted_avg_accuracy": float(avg_acc),
            "total_samples": int(stats["sum_samples"])
        }

    # Pick global winner by highest weighted average Accuracy
    valid_models = {m: v for m, v in combined.items() if not np.isnan(v["weighted_avg_accuracy"])}
    if valid_models:
        global_winner_name = max(valid_models.keys(), key=lambda m: valid_models[m]["weighted_avg_accuracy"])
        global_winner_acc = valid_models[global_winner_name]["weighted_avg_accuracy"]
        print("\n=== Overall Model Winner (all AssetTypes combined) ===")
        print("Model    Weighted_Avg_Accuracy%   Total_Samples")
        for m, v in combined.items():
            print(f"{m:<7} {v['weighted_avg_accuracy']:<24.2f} {v['total_samples']}")
        print(f"\n🏆 GLOBAL WINNER: {global_winner_name} with weighted Accuracy={global_winner_acc:.2f}%")
    else:
        global_winner_name = None
        global_winner_acc = None
        print("\nNo valid models for global winner calculation.")

    # Save JSON report
    if not args.no_summary:
        summary = {
            "results": results,
            "combined": combined,
            "global_winner": {
                "name": global_winner_name,
                "weighted_avg_accuracy": global_winner_acc
            }
        }
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary → {args.summary_path}")

    print("\n🎯 Evaluation complete.")


if __name__ == "__main__":
    main()
