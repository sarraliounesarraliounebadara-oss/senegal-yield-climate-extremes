#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-model robustness of projected yield changes.
Each future model mean is compared with that same model's 1986-2013 RF historical baseline.
No plotting.
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

PERIOD_FILES = {"2036-2065": "near_future", "2071-2100": "far_future"}
SCENARIOS = ["ssp245", "ssp585"]
SCEN_LABEL = {"ssp245": "SSP245", "ssp585": "SSP585"}

detail_rows = []

for crop in CROPS:
    hist = filter_hist(pd.read_csv(RF_H_DIR / crop / f"RF_{crop}_historical.csv"))
    hist_mean = hist.groupby("model")["pred_anomaly"].mean()

    for period, code in PERIOD_FILES.items():
        fut = pd.read_csv(RF_H_DIR / crop / f"RF_{crop}_{code}.csv")
        for scenario in SCENARIOS:
            sub = fut[fut["scenario"] == scenario]
            fut_mean = sub.groupby("model")["pred_anomaly"].mean()
            common = sorted(set(hist_mean.index) & set(fut_mean.index))
            for model in common:
                detail_rows.append({
                    "Crop": CROP_LABELS[crop], "Period": period, "Scenario": SCEN_LABEL[scenario],
                    "Model": model, "Delta_yield_pct": float(fut_mean[model] - hist_mean[model]),
                })

detail = pd.DataFrame(detail_rows)
summary_rows = []
for (crop, period, scenario), g in detail.groupby(["Crop", "Period", "Scenario"], sort=False):
    v = g["Delta_yield_pct"].to_numpy(float)
    nneg = int(np.sum(v < 0)); npos = int(np.sum(v > 0)); nzero = int(np.sum(v == 0))
    majority = "negative" if nneg > npos else "positive" if npos > nneg else "tie"
    agreement = max(nneg, npos, nzero) / len(v)
    summary_rows.append({
        "Crop": crop, "Period": period, "Scenario": scenario, "N_models": len(v),
        "Mean_delta_yield_pct": np.mean(v), "Median_delta_yield_pct": np.median(v),
        "Q25": np.quantile(v, 0.25), "Q75": np.quantile(v, 0.75),
        "Min": np.min(v), "Max": np.max(v), "N_negative": nneg, "N_positive": npos,
        "N_zero": nzero, "Majority_sign": majority, "Majority_sign_fraction": agreement,
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "Table_S6_future_yield_ensemble_robustness.csv", index=False)
detail.to_csv(OUT_DIR / "Table_S6b_future_yield_model_detail.csv", index=False)
print(summary.to_string(index=False))
