#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Maize >=80% record-completeness sensitivity test. No plotting."""

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
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer

crop = "maize"
df = pd.read_csv(OBS_DIR / "maize_extremes.csv")
df = to_float32_like_rf(df)
df = df[df["year"].between(HIST_START, HIST_END) & df["yield_anomaly"].notna()].reset_index(drop=True)

n_years_total = HIST_END - HIST_START + 1
counts = df.groupby("department")["year"].nunique().sort_index()
comp = pd.DataFrame({"department": counts.index, "N_years": counts.values})
comp["Completeness_pct"] = 100 * comp["N_years"] / n_years_total
comp["Included_ge80pct"] = comp["Completeness_pct"] >= 80.0
comp.to_csv(OUT_DIR / "Table_S5a_maize_department_completeness.csv", index=False)
keep = comp.loc[comp["Included_ge80pct"], "department"].tolist()

# Full-sample baseline SHAP from the production Hargreaves RF output.
base = pd.read_csv(RF_H_DIR / crop / f"RF_{crop}_ANACIM_historical.csv")
base = base[base["year"].between(HIST_START, HIST_END)].copy()
base_imp = pd.Series({
    feat: float(base[f"shap_{feat}"].abs().mean()) for feat in FEATURES
})
base_rank = base_imp.rank(ascending=False, method="min")
base_met = metrics(base["yield_anomaly"], base["pred_anomaly"])

sub = df[df["department"].isin(keep)].copy()
imp = SimpleImputer(strategy="median")
X = pd.DataFrame(imp.fit_transform(sub[FEATURES]), columns=FEATURES, index=sub.index)
y = sub["yield_anomaly"].astype(float)
model = rf_model(); model.fit(X, y)
pred = model.predict(X)
sub_met = metrics(y, pred)
sv = np.asarray(shap.TreeExplainer(model).shap_values(X))
sub_imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES)
sub_rank = sub_imp.rank(ascending=False, method="min")

p_r = pearsonr(base_imp.values, sub_imp.reindex(base_imp.index).values).statistic
s_r = spearmanr(base_imp.values, sub_imp.reindex(base_imp.index).values).statistic

rows = [
    {"Sample": "All maize departments", "N_departments": base["department"].nunique(), **base_met,
     "SPEI3_rank": int(base_rank["SPEI3_Hargreaves"]),
     "SPEI3_mean_abs_SHAP": base_imp["SPEI3_Hargreaves"],
     "SHAP_Pearson_r_vs_full": 1.0, "SHAP_Spearman_rho_vs_full": 1.0},
    {"Sample": "Departments with >=80% completeness", "N_departments": len(keep), **sub_met,
     "SPEI3_rank": int(sub_rank["SPEI3_Hargreaves"]),
     "SPEI3_mean_abs_SHAP": sub_imp["SPEI3_Hargreaves"],
     "SHAP_Pearson_r_vs_full": p_r, "SHAP_Spearman_rho_vs_full": s_r},
]
pd.DataFrame(rows).to_csv(OUT_DIR / "Table_S5_maize_missingness_sensitivity.csv", index=False)

shap_table = pd.DataFrame({
    "Feature": [x.replace("SPEI3_Hargreaves", "SPEI3") for x in FEATURES],
    "Full_mean_abs_SHAP": base_imp.values,
    "Complete_subset_mean_abs_SHAP": sub_imp.reindex(base_imp.index).values,
    "Full_rank": base_rank.astype(int).values,
    "Complete_subset_rank": sub_rank.reindex(base_imp.index).astype(int).values,
})
shap_table.to_csv(OUT_DIR / "Table_S5b_maize_missingness_SHAP.csv", index=False)

print(pd.DataFrame(rows).to_string(index=False))
