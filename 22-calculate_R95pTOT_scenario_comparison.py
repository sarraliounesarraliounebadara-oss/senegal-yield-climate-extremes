#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R95pTOT SSP2-4.5 vs SSP5-8.5 model-wise comparison.
Uses RAW climate tables and the fixed 1984-2013 climate reference used for the
R95pTOT threshold / climate baseline in this sensitivity calculation.
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

REF_START, REF_END = 1984, 2013
PERIOD_FILES = {"2036-2065": "near_future", "2071-2100": "far_future"}
rows = []

for crop in CROPS:
    hist = pd.read_csv(CMIP6_DIR / f"{crop}_historical_extremes.csv")
    hist = hist[hist["year"].between(REF_START, REF_END)]
    hist_mean = hist.groupby("model")["R95pTOT"].mean()

    for period, code in PERIOD_FILES.items():
        fut = pd.read_csv(CMIP6_DIR / f"{crop}_{code}_extremes.csv")
        by_scenario = {}
        for scenario in ["ssp245", "ssp585"]:
            means = fut[fut["scenario"] == scenario].groupby("model")["R95pTOT"].mean()
            common = sorted(set(hist_mean.index) & set(means.index))
            by_scenario[scenario] = pd.Series({m: means[m] - hist_mean[m] for m in common})

        common = sorted(set(by_scenario["ssp245"].index) & set(by_scenario["ssp585"].index))
        d245 = by_scenario["ssp245"].reindex(common).to_numpy(float)
        d585 = by_scenario["ssp585"].reindex(common).to_numpy(float)
        rows.append({
            "Crop": CROP_LABELS[crop], "Period": period, "N_models": len(common),
            "Mean_delta_SSP245": np.mean(d245), "Median_delta_SSP245": np.median(d245),
            "Mean_delta_SSP585": np.mean(d585), "Median_delta_SSP585": np.median(d585),
            "N_models_delta245_gt_delta585": int(np.sum(d245 > d585)),
            "Fraction_delta245_gt_delta585": float(np.mean(d245 > d585)),
            "Mean_difference_delta245_minus_delta585": float(np.mean(d245 - d585)),
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "Table_S7_R95pTOT_scenario_change_comparison.csv", index=False)
print(out.to_string(index=False))
