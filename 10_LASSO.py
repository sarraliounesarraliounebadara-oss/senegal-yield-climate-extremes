# =========================================
# LASSO + LOYO + YIELD RECONSTRUCTION
# =========================================

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
import unicodedata

# =========================================
# CLEAN
# =========================================
def clean_dep(x):
    if pd.isna(x):
        return x
    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("utf-8")
    x = x.replace("-", " ").replace("_", " ")
    return x

# =========================================
# PATHS
# =========================================
BASE = "/media/sarr/01DC5DE9D15E8CF0/ml_datasets"
OUT  = "/media/sarr/01DC5DE9D15E8CF0/results_agro/LASSO_LOYO"

os.makedirs(OUT, exist_ok=True)

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

    print(f"\n===== {crop.upper()} =====")

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

    df["department"] = df["department"].apply(clean_dep)

    df = df.dropna(subset=features + [target])

    if df.empty:
        continue

    X = df[features]
    y = df[target]

    if "year" not in df.columns:
        continue

    groups = df["year"].astype(int)
    logo = LeaveOneGroupOut()

    # =========================
    # LOYO
    # =========================
    preds, y_true = [], []

    for train_idx, test_idx in logo.split(X, y, groups):

        scaler_cv = StandardScaler()
        X_train_s = scaler_cv.fit_transform(X.iloc[train_idx])
        X_test_s  = scaler_cv.transform(X.iloc[test_idx])

        model_cv = LassoCV(
            alphas=np.logspace(-6, -1, 100),
            max_iter=5000,
            cv=10,
            random_state=42
        )

        model_cv.fit(X_train_s, y.iloc[train_idx])
        preds.extend(model_cv.predict(X_test_s))
        y_true.extend(y.iloc[test_idx])

    print("LOYO R²:", round(r2_score(y_true, preds), 3))

    # =========================
    # FINAL MODEL
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LassoCV(
        alphas=np.logspace(-9, -1, 100),
        max_iter=5000,
        cv=10,
        random_state=42
    )

    model.fit(X_scaled, y)

    print("Intercept:", model.intercept_)

    coefs = model.coef_
    means = scaler.mean_
    scales = scaler.scale_

    # domaine 
    X_min = X.min()
    X_max = X.max()

    # =========================
    # APPLY
    # =========================
    for period in ["historical","near_future","far_future"]:

        file = f"{BASE}/{crop}/{period}.csv"
        if not os.path.exists(file):
            continue

        dfp = pd.read_csv(file)
        dfp.columns = [c.strip() for c in dfp.columns]

        for col in base_cols:
            if col in dfp.columns:
                dfp[col] = pd.to_numeric(dfp[col], errors="coerce")

        # interactions
        dfp["heat_drought"] = dfp["TXx"] * dfp["CDD"]
        dfp["extreme_rain"] = dfp["RX1day"] * dfp["R95pTOT"]
        dfp["heat_intensity"] = dfp["TX35"] * dfp["CDD"]
        dfp["drought_severity"] = dfp["CDD"] * dfp["SPEI3"]
        dfp["wet_extreme"] = dfp["RX1day"] * dfp["PRtotal"]
        dfp["heat_water_balance"] = dfp["TXx"] * dfp["SPEI3"]

        dfp["department"] = dfp["department"].apply(clean_dep)

        mask = dfp[features].notna().all(axis=1)

        # domaine corrigé (important)
        tol = 0.2
        mask_domain = (
            (dfp[features] >= (X_min - tol * (X_max - X_min))) &
            (dfp[features] <= (X_max + tol * (X_max - X_min)))
        ).all(axis=1)

        mask = mask & mask_domain

        if mask.sum() == 0:
            continue

        Xg = dfp.loc[mask, features]
        Xg_scaled = scaler.transform(Xg)

        # =========================
        # PREDICTION
        # =========================
        dfp.loc[mask, "prediction"] = model.predict(Xg_scaled)

        # =========================
        # YIELD 
        # =========================
        if "yield_trend" in dfp.columns:
            valid_trend = dfp["yield_trend"].notna()
            final_mask = mask & valid_trend

            dfp.loc[final_mask, "pred_yield"] = (
                dfp.loc[final_mask, "yield_trend"] *
                (1 + dfp.loc[final_mask, "prediction"] / 100)
            )

        # =========================
        # CONTRIBUTIONS 
        # =========================
        for i, col in enumerate(features):

            if scales[i] == 0:
                continue

            beta_real = coefs[i] / scales[i]
            centered = dfp.loc[mask, col] - means[i]

            dfp.loc[mask, f"coef_{col}"] = beta_real
            dfp.loc[mask, f"contrib_{col}"] = centered * beta_real

        print("pred anomaly range:",
              np.nanmin(dfp["prediction"]),
              np.nanmax(dfp["prediction"]))

        out_file = f"{crop_out}/LASSO_{crop}_{period}.csv"
        dfp.to_csv(out_file, index=False)

        print("saved:", out_file)

print("\nLASSO FINAL ALIGN RF DONE")
