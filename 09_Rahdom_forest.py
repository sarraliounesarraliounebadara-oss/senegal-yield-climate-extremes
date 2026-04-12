# =========================================
# RF + LOYO + SHAP + RECONSTRUCTION YIELD (FINAL CLEAN)
# =========================================

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
import shap

# =========================================
# PATHS
# =========================================
BASE = "/media/sarr/01DC5DE9D15E8CF0/ml_datasets"
OUT  = "/media/sarr/01DC5DE9D15E8CF0/results_agro/RF_LOYO"

os.makedirs(OUT, exist_ok=True)

# =========================================
# FEATURES
# =========================================
features = [
    "TXx","TX35","RX1day","CDD","R95pTOT","PRtotal","SPEI3",
    "heat_drought","extreme_rain","heat_intensity",
    "drought_severity","wet_extreme","heat_water_balance"
]

target = "yield_anomaly"

crops = ["maize","millet","sorghum","groundnut"]

# =========================================
# LOOP
# =========================================
for crop in crops:

    print("\n=====", crop.upper(), "=====")

    crop_out = os.path.join(OUT, crop)
    os.makedirs(crop_out, exist_ok=True)

    file_hist = f"{BASE}/{crop}/historical.csv"

    if not os.path.exists(file_hist):
        print("missing:", file_hist)
        continue

    df = pd.read_csv(file_hist)
    df.columns = [c.strip() for c in df.columns]

    # =========================
    # CONVERSION
    # =========================
    base_cols = ["TXx","TX35","RX1day","CDD","R95pTOT","PRtotal","SPEI3","yield_anomaly"]

    for col in base_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # INTERACTIONS
    # =========================
    df["heat_drought"] = df["TXx"] * df["CDD"]
    df["extreme_rain"] = df["RX1day"] * df["R95pTOT"]
    df["heat_intensity"] = df["TX35"] * df["CDD"]
    df["drought_severity"] = df["CDD"] * df["SPEI3"]
    df["wet_extreme"] = df["RX1day"] * df["PRtotal"]
    df["heat_water_balance"] = df["TXx"] * df["SPEI3"]

    # =========================
    # CHECK
    # =========================
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        print("colonnes manquantes:", missing)
        continue

    # =========================
    # TRAIN DATA
    # =========================
    train_mask = df[features + [target]].notna().all(axis=1)

    X = df.loc[train_mask, features]
    y = df.loc[train_mask, target]

    if X.empty:
        print("no training data")
        continue

    # domaine 
    X_min = X.min()
    X_max = X.max()

    # =========================
    # GROUPS
    # =========================
    if "year" not in df.columns:
        print("year manquant")
        continue

    groups = df.loc[train_mask, "year"].astype(int)
    logo = LeaveOneGroupOut()

    # =========================
    # LOYO
    # =========================
    preds, y_true = [], []

    for train_idx, test_idx in logo.split(X, y, groups):

        model = RandomForestRegressor(
            n_estimators=700,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.extend(model.predict(X.iloc[test_idx]))
        y_true.extend(y.iloc[test_idx])

    print("LOYO R²:", round(r2_score(y_true, preds), 3))

    # =========================
    # FINAL MODEL
    # =========================
    model = RandomForestRegressor(
        n_estimators=700,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    print("y std:", y.std())
    print("pred std:", np.std(model.predict(X)))

    explainer = shap.TreeExplainer(model)

    # =========================================
    # APPLY
    # =========================================
    for period in ["historical","near_future","far_future"]:

        f = f"{BASE}/{crop}/{period}.csv"

        if not os.path.exists(f):
            continue

        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]

        # conversion
        for col in base_cols:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")

        # =========================
        # INTERACTIONS
        # =========================
        d["heat_drought"] = d["TXx"] * d["CDD"]
        d["extreme_rain"] = d["RX1day"] * d["R95pTOT"]
        d["heat_intensity"] = d["TX35"] * d["CDD"]
        d["drought_severity"] = d["CDD"] * d["SPEI3"]
        d["wet_extreme"] = d["RX1day"] * d["PRtotal"]
        d["heat_water_balance"] = d["TXx"] * d["SPEI3"]

        # =========================
        # MASKS
        # =========================
        mask_features = d[features].notna().all(axis=1)

        if period == "historical":
            mask_target = d["yield_anomaly"].notna()
            mask = mask_features & mask_target
        else:
            mask = mask_features

        # DOMAIN (plus robuste)
        tol = 0.2
        mask_domain = (
            (d[features] >= (X_min - tol * (X_max - X_min))) &
            (d[features] <= (X_max + tol * (X_max - X_min)))
        ).all(axis=1)

        mask = mask & mask_domain

        d["pred_anomaly"] = np.nan
        d["pred_yield"] = np.nan

        if mask.sum() == 0:
            continue

        X_pred = d.loc[mask, features]

        # =========================
        # PREDICTION
        # =========================
        d.loc[mask, "pred_anomaly"] = model.predict(X_pred)

        # reconstruction yield
        if "yield_trend" in d.columns:
            valid_trend = d["yield_trend"].notna()
            final_mask = mask & valid_trend

            d.loc[final_mask, "pred_yield"] = (
                d.loc[final_mask, "yield_trend"] *
                (1 + d.loc[final_mask, "pred_anomaly"] / 100)
            )

        print("pred anomaly range:",
              np.nanmin(d["pred_anomaly"]),
              np.nanmax(d["pred_anomaly"]))

        # =========================
        # SHAP (corrigé)
        # =========================
        sample_size = min(1000, len(X_pred))
        X_shap = X_pred.sample(n=sample_size, random_state=42)

        shap_vals = explainer(X_shap)
        shap_df = pd.DataFrame(shap_vals.values, columns=features)
        shap_df.index = X_shap.index

        for col in features:
            d.loc[X_shap.index, f"shap_{col}"] = shap_df[col]

        # =========================
        # IMPORTANCE
        # =========================
        importance = shap_df.abs().mean().sort_values(ascending=False)

        print("\nSHAP importance:")
        print(importance)

        # =========================
        # SAVE
        # =========================
        out = f"{crop_out}/RF_{crop}_{period}.csv"
        d.to_csv(out, index=False)

        print("saved:", out)

print("\nRF + SHAP + YIELD DONE")
