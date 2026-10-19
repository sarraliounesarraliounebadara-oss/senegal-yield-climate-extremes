#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Out-of-domain (OOD) diagnostic for all seven RF predictors.
Uses RAW pre-clipping CMIP6 crop tables, not RF projection files.
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

PREDICTORS = {
    "TXx": "TXx",
    "TNx": "TNx",
    "RX5day": "RX5day",
    "R95pTOT": "R95pTOT",
    "CDD": "CDD",
    "PRtot": "PRtot",
    "SPEI3": "SPEI3_Hargreaves",
}
PERIOD_FILES = {
    "2036-2065": "near_future",
    "2071-2100": "far_future",
}
SCENARIOS = ["ssp245", "ssp585"]
SCEN_LABEL = {"ssp245": "SSP245", "ssp585": "SSP585"}

training_rows = []
ood_rows = []
model_rows = []

for crop in CROPS:
    obs = pd.read_csv(OBS_DIR / f"{crop}_extremes.csv")
    obs = obs[obs["year"].between(HIST_START, HIST_END) & obs["yield_anomaly"].notna()].copy()

    ranges = {}
    for label, col in PREDICTORS.items():
        s = pd.to_numeric(obs[col], errors="coerce").dropna()
        lo, hi = float(s.min()), float(s.max())
        ranges[label] = (lo, hi)
        training_rows.append({
            "Crop": CROP_LABELS[crop],
            "Predictor": label,
            "N_training_valid": len(s),
            "Hist_min": lo,
            "Hist_max": hi,
        })

    for period, code in PERIOD_FILES.items():
        fut = pd.read_csv(CMIP6_DIR / f"{crop}_{code}_extremes.csv")
        for scenario in SCENARIOS:
            sub = fut[fut["scenario"] == scenario].copy()
            for label, col in PREDICTORS.items():
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                lo, hi = ranges[label]
                below = 100.0 * float((vals < lo).mean()) if len(vals) else np.nan
                above = 100.0 * float((vals > hi).mean()) if len(vals) else np.nan
                ood_rows.append({
                    "Crop": CROP_LABELS[crop], "Period": period, "Scenario": SCEN_LABEL[scenario],
                    "Predictor": label, "N_valid": len(vals), "Hist_min": lo, "Hist_max": hi,
                    "Pct_below_hist_min": below, "Pct_above_hist_max": above,
                    "Pct_OOD": below + above,
                })

                for model, g in sub.groupby("model"):
                    v = pd.to_numeric(g[col], errors="coerce").dropna()
                    pct = 100.0 * float(((v < lo) | (v > hi)).mean()) if len(v) else np.nan
                    model_rows.append({
                        "Crop": CROP_LABELS[crop], "Period": period, "Scenario": SCEN_LABEL[scenario],
                        "Predictor": label, "Model": model, "N_valid": len(v), "Pct_OOD": pct,
                    })

training = pd.DataFrame(training_rows)
ood = pd.DataFrame(ood_rows)
model_detail = pd.DataFrame(model_rows)
wide = ood.pivot_table(index=["Crop", "Period", "Scenario"], columns="Predictor", values="Pct_OOD").reset_index()
wide.columns.name = None
ordered = ["Crop", "Period", "Scenario", "SPEI3", "TXx", "TNx", "RX5day", "R95pTOT", "CDD", "PRtot"]
wide = wide[[c for c in ordered if c in wide.columns]]

training.to_csv(OUT_DIR / "Table_S3a_training_ranges.csv", index=False)
ood.to_csv(OUT_DIR / "Table_S3_OOD_all_predictors.csv", index=False)
model_detail.to_csv(OUT_DIR / "Table_S3b_OOD_model_detail.csv", index=False)
wide.to_csv(OUT_DIR / "Table_S3_OOD_wide.csv", index=False)

print(wide.to_string(index=False))
