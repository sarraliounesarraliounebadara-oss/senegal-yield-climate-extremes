#!/usr/bin/env python3

# =========================================
# RF + LOYO + SHAP + RECONSTRUCTION YIELD
# VERSION ALIGNEMENT STRICT 1:1 & MÉTRIQUES COMPLÈTES
# =========================================

import os
import gc
import shap
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# =========================================
# PATHS
# =========================================

BASE = "/media/sarr/01DC5DE9D15E8CF0"

OBS_DIR = (
    f"{BASE}/departments/climate_data/"
    f"extremes/crop_tables"
)

CMIP6_DIR = (
    f"{BASE}/departments/climate_data/"
    f"extremes/cmip6_crop_tables"
)

OUT = (
    f"{BASE}/results_agro/"
    f"RF_LOYO_CONSERVATIVE"
)

os.makedirs(OUT, exist_ok=True)

# =========================================
# FEATURES
# =========================================

features = [
    "TXx",
    "TNx",
    "RX5day",
    "R95pTOT",
    "CDD",
    "PRtot",
    "SPEI3",
]

target = "yield_anomaly"

crops = [
    "sorghum",
    "maize",
    "millet",
    "groundnut"
]

# =========================================
# MEMORY OPT
# =========================================

def optimize_dataframe(df):
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype("float32")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = df[col].astype("int32")
    return df

# =========================================
# LOOP
# =========================================

for crop in crops:
    print("\n===================================")
    print(crop.upper())
    print("===================================")

    crop_out = os.path.join(OUT, crop)
    os.makedirs(crop_out, exist_ok=True)

    # =====================================
    # OBS FILE
    # =====================================
    file_hist = f"{OBS_DIR}/{crop}_extremes.csv"

    if not os.path.exists(file_hist):
        print("\nMissing:")
        print(file_hist)
        continue

    # =====================================
    # LOAD
    # =====================================
    df = pd.read_csv(file_hist)
    df.columns = [c.strip() for c in df.columns]

    if "month" in df.columns:
        df.drop(columns=["month"], inplace=True)

    df = optimize_dataframe(df)

    # =====================================
    # NUMERIC
    # =====================================
    base_cols = ["TXx", "TNx", "R95pTOT", "RX5day", "CDD", "PRtot", "SPEI3"]
    numeric_cols = base_cols + [target]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # =====================================
    # CHECK COLUMNS
    # =====================================
    missing = [c for c in (features + [target]) if c not in df.columns]

    if missing:
        print("\nMissing columns:")
        print(missing)
        continue

    # =====================================
    # ALIGNEMENT STRICT 1:1
    # =====================================
    # On supprime les lignes sans observation réelle pour garantir l'égalité parfaite
    df = df.dropna(subset=[target, "year"]).reset_index(drop=True)

    # =====================================
    # PREPARATION & IMPUTATION CLIMATIQUE
    # =====================================
    X_raw = df[features].copy()
    y = df[target].copy()
    groups = df["year"]

    # Remplissage des indices climatiques manquants par leur médiane
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)
    X = pd.DataFrame(X_imputed, columns=features, index=df.index)

    print(f"ℹ️ Alignement : {len(df)} lignes observées appairées à {len(df)} prédictions.")

    # =====================================
    # LOYO (VALIDATION CROISÉE)
    # =====================================
    logo = LeaveOneGroupOut()
    preds_loyo = np.full(len(y), np.nan)

    for train_idx, test_idx in logo.split(X, y, groups):
        model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=12,
            max_leaf_nodes=150,
            min_samples_split=4,
            min_samples_leaf=2,
            min_weight_fraction_leaf=0.0,
            max_features=0.8,
            criterion="squared_error",
            min_impurity_decrease=1e-7,
            ccp_alpha=0.001,
            bootstrap=True,
            oob_score=True,
            max_samples=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds_loyo[test_idx] = model.predict(X.iloc[test_idx])

    # RENTRÉE DES SCORES LOYO
    r2_loyo = r2_score(y, preds_loyo)
    corr_loyo = np.corrcoef(y, preds_loyo)[0, 1]

    print("\nLOYO R²:", round(r2_loyo, 3))
    print("LOYO Corr:", round(corr_loyo, 3))

    # =====================================
    # FINAL MODEL & RECONSTRUCTION
    # =====================================
    model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=12,
        max_leaf_nodes=150,
        min_samples_split=4,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0.0,
        max_features=0.8,
        criterion="squared_error",
        min_impurity_decrease=1e-7,
        ccp_alpha=0.001,
        bootstrap=True,
        oob_score=True,
        max_samples=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    model.fit(X, y)
    hist_pred = model.predict(X)

    # RENTRÉE DES SCORES HISTORIQUES (RÉTABLIS)
    r2_hist = r2_score(y, hist_pred)
    corr_hist = np.corrcoef(y, hist_pred)[0, 1]

    print("\nHistorical R²:", round(r2_hist, 3))
    print("Historical Corr:", round(corr_hist, 3))

    # =====================================
    # SHAP HISTORICAL (RÉTABLI ET AFFICHÉ)
    # =====================================
    explainer = shap.TreeExplainer(model)
    shap_vals_obs = explainer.shap_values(X)
    shap_obs_df = pd.DataFrame(shap_vals_obs, columns=features, index=X.index)

    # Affichage de l'importance SHAP historique
    importance_obs = shap_obs_df.abs().mean().sort_values(ascending=False)
    print("\nHistorical SHAP importance:")
    print(importance_obs)

    # =====================================
    # SAVE OBS
    # =====================================
    obs_hist = df.copy()
    obs_hist["pred_anomaly"] = hist_pred
    obs_hist["pred_anomaly_loyo"] = preds_loyo 

    for col in features:
        obs_hist[f"shap_{col}"] = shap_obs_df[col]

    hist_out = f"{crop_out}/RF_{crop}_ANACIM_historical.csv"
    obs_hist.to_csv(hist_out, index=False)
    print(f"\nSaved historical file: {hist_out}")

    # =====================================
    # APPLY CMIP6
    # =====================================
    periods = ["historical", "near_future", "far_future"]

    for period in periods:
        file_period = f"{CMIP6_DIR}/{crop}_{period}_extremes.csv"

        if not os.path.exists(file_period):
            print("\nMissing:")
            print(file_period)
            continue

        print(f"\nLoading {period}...")
        d = pd.read_csv(file_period)
        d.columns = [c.strip() for c in d.columns]
        d = optimize_dataframe(d)

        for col in base_cols:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")

        for col in features:
            low = X[col].quantile(0.01)
            high = X[col].quantile(0.99)
            d[col] = d[col].clip(lower=low, upper=high)

        # Imputation CMIP6
        X_pred_imputed = imputer.transform(d[features])
        X_pred = pd.DataFrame(X_pred_imputed, columns=features, index=d.index)

        # Prédiction
        pred = model.predict(X_pred)
        pred = np.clip(pred, y.quantile(0.01), y.quantile(0.99))
        d["pred_anomaly"] = pred

        # SHAP Future
        shap_vals = explainer.shap_values(X_pred)
        shap_df = pd.DataFrame(shap_vals, columns=features, index=X_pred.index)

        for col in features:
            d[f"shap_{col}"] = shap_df[col]

        # Importance SHAP Future
        importance = shap_df.abs().mean().sort_values(ascending=False)
        print(f"\nSHAP importance for {period}:")
        print(importance)

        # Sauvegarde
        out = f"{crop_out}/RF_{crop}_{period}.csv"
        d.to_csv(out, index=False)
        print(f"Saved projection file: {out}")

        del d
        del shap_df
        gc.collect()

print("\n===================================")
print("RF ALIGNEMENT 1:1 + TOUTES MÉTRIQUES TERMINÉ")
print("===================================")
