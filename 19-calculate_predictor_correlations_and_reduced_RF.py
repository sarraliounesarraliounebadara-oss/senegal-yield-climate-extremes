#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictor dependence + reduced-predictor Random Forest robustness.
No plotting.

1) Pearson correlation matrices for each crop over 1986-2013.
2) Identify pairs with |r| >= 0.70.
3) For the systematic RX5day-PRtot pair, remove the lower-SHAP member.
4) Refit the same prespecified RF and recompute fitted metrics + SHAP.
"""

# ============================================================
# COMMON SETTINGS — standalone script
# ============================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CROPS = ["maize", "millet", "sorghum", "groundnut"]
CROP_LABELS = {
    "maize": "Maize",
    "millet": "Millet",
    "sorghum": "Sorghum",
    "groundnut": "Groundnut",
}
FEATURES = ["TXx", "TNx", "RX5day", "R95pTOT", "CDD", "PRtot", "SPEI3_Hargreaves"]
FEATURE_LABELS = ["TXx", "TNx", "RX5day", "R95pTOT", "CDD", "PRtot", "SPEI3"]
TARGET = "yield_anomaly"
HIST_START = 1986
HIST_END = 2013

BASE = Path(os.environ.get("BASE", "/mnt/disque"))
OBS_DIR = Path(os.environ.get("OBS_DIR", BASE / "departments/climate_data/extremes/crop_tables"))
CMIP6_DIR = Path(os.environ.get("CMIP6_DIR", BASE / "departments/climate_data/extremes/cmip6_crop_tables"))
RF_H_DIR = Path(os.environ.get("RF_H_DIR", BASE / "results_agro/RF_LOYO_HARGREAVES"))
if not RF_H_DIR.exists():
    alt = BASE / "results_agro/RF_LOYO_CONSERVATIVE"
    if alt.exists():
        RF_H_DIR = alt
RF_M_DIR = Path(os.environ.get("RF_M_DIR", BASE / "results_agro/RF_LOYO_Makkink"))
OUT_DIR = Path(os.environ.get("REVISION_OUT", BASE / "results_agro/reviewer_revision_tables"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rf_model() -> RandomForestRegressor:
    return RandomForestRegressor(
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
        verbose=0,
    )


def filter_hist(df: pd.DataFrame, start: int = HIST_START, end: int = HIST_END) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns:
        out = out[out["year"].between(start, end)]
    return out.copy()


def metrics(y, pred) -> dict:
    tmp = pd.DataFrame({"y": y, "pred": pred}).dropna()
    yy = tmp["y"].to_numpy(float)
    pp = tmp["pred"].to_numpy(float)
    r = np.corrcoef(yy, pp)[0, 1] if len(yy) > 1 else np.nan
    return {
        "N": len(yy),
        "R2": r2_score(yy, pp),
        "Pearson_r": r,
        "RMSE": mean_squared_error(yy, pp) ** 0.5,
        "MAE": mean_absolute_error(yy, pp),
        "Mean_bias": float(np.mean(pp - yy)),
    }


def to_float32_like_rf(df: pd.DataFrame) -> pd.DataFrame:
    """Mimic the dtype reduction used in the RF production script."""
    df = df.copy()
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype("int32")
    return df



import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer

corr_rows = []
high_rows = []
reduced_rows = []

for crop in CROPS:
    df = pd.read_csv(OBS_DIR / f"{crop}_extremes.csv")
    df = to_float32_like_rf(df)
    df = df[df["year"].between(HIST_START, HIST_END) & df["yield_anomaly"].notna()].reset_index(drop=True)

    corr = df[FEATURES].corr(method="pearson")
    for a in FEATURES:
        for b in FEATURES:
            corr_rows.append({"Crop": CROP_LABELS[crop], "Predictor_1": a.replace("SPEI3_Hargreaves", "SPEI3"),
                              "Predictor_2": b.replace("SPEI3_Hargreaves", "SPEI3"), "r": corr.loc[a, b]})

    seen = set()
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i+1:]:
            r = corr.loc[a, b]
            if np.isfinite(r) and abs(r) >= 0.70:
                high_rows.append({"Crop": CROP_LABELS[crop],
                                  "Predictor_1": a.replace("SPEI3_Hargreaves", "SPEI3"),
                                  "Predictor_2": b.replace("SPEI3_Hargreaves", "SPEI3"), "r": r})

    # Baseline SHAP importance from the production RF output, restricted to manuscript years.
    base = pd.read_csv(RF_H_DIR / crop / f"RF_{crop}_ANACIM_historical.csv")
    base = base[base["year"].between(HIST_START, HIST_END)].copy()
    shap_imp = {}
    for feat in FEATURES:
        scol = f"shap_{feat}"
        shap_imp[feat] = float(base[scol].abs().mean())

    pair = ["RX5day", "PRtot"]
    removed = min(pair, key=lambda x: shap_imp[x])
    reduced_features = [x for x in FEATURES if x != removed]

    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(df[reduced_features]), columns=reduced_features, index=df.index)
    y = df["yield_anomaly"].astype(float)
    model = rf_model()
    model.fit(X, y)
    pred = model.predict(X)
    met = metrics(y, pred)

    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(X))
    red_imp = pd.Series(np.abs(sv).mean(axis=0), index=reduced_features)
    spei = "SPEI3_Hargreaves"
    base_rank = pd.Series(shap_imp).rank(ascending=False, method="min")[spei]
    red_rank = red_imp.rank(ascending=False, method="min")[spei]
    r_pair = corr.loc["RX5day", "PRtot"]

    reduced_rows.append({
        "Crop": CROP_LABELS[crop],
        "Strong_pair": f"RX5day-PRtot (r={r_pair:.2f})",
        "Removed_predictor": removed,
        "R2_reduced": met["R2"], "RMSE_reduced": met["RMSE"], "MAE_reduced": met["MAE"],
        "Bias_reduced": met["Mean_bias"],
        "SPEI3_rank_baseline": int(base_rank), "SPEI3_rank_reduced": int(red_rank),
        "SPEI3_SHAP_baseline": shap_imp[spei], "SPEI3_SHAP_reduced": float(red_imp[spei]),
    })

pd.DataFrame(corr_rows).to_csv(OUT_DIR / "Table_S4_full_correlation_matrices_long.csv", index=False)
pd.DataFrame(high_rows).to_csv(OUT_DIR / "Table_S4a_high_correlations_1986_2013.csv", index=False)
pd.DataFrame(reduced_rows).to_csv(OUT_DIR / "Table_S4_reduced_RF_robustness.csv", index=False)

print(pd.DataFrame(high_rows).to_string(index=False))
print("\nReduced RF robustness")
print(pd.DataFrame(reduced_rows).to_string(index=False))
