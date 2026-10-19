#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hargreaves-versus-Makkink sensitivity tables used in the revision.
Works from the two RF output trees. No plotting.
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
from scipy.stats import pearsonr, spearmanr

BASE_FEATURES = ["TXx", "TNx", "RX5day", "R95pTOT", "CDD", "PRtot"]
PERIOD_FILES = {"2036-2065": "near_future", "2071-2100": "far_future"}
SCENARIOS = ["ssp245", "ssp585"]
SCEN_LABEL = {"ssp245": "SSP245", "ssp585": "SSP585"}

summary_rows, hist_metric_rows, shap_rows = [], [], []
model_detail_rows = []

for crop in CROPS:
    h = filter_hist(pd.read_csv(RF_H_DIR / crop / f"RF_{crop}_ANACIM_historical.csv"))
    m = filter_hist(pd.read_csv(RF_M_DIR / crop / f"RF_{crop}_ANACIM_historical.csv"))

    for pet, df in [("Hargreaves", h), ("Makkink", m)]:
        met = metrics(df["yield_anomaly"], df["pred_anomaly"])
        hist_metric_rows.append({"Crop": CROP_LABELS[crop], "PET": pet, **met})

    join = h[["department", "year", "SPEI3_Hargreaves"]].merge(
        m[["department", "year", "SPEI3_Makkink"]], on=["department", "year"], how="inner"
    ).dropna()
    spei_corr = join[["SPEI3_Hargreaves", "SPEI3_Makkink"]].corr().iloc[0, 1]

    h_imp, m_imp = [], []
    for feat in BASE_FEATURES + ["SPEI3"]:
        hcol = f"shap_{feat}" if feat != "SPEI3" else "shap_SPEI3_Hargreaves"
        mcol = f"shap_{feat}" if feat != "SPEI3" else "shap_SPEI3_Makkink"
        hv = float(h[hcol].abs().mean())
        mv = float(m[mcol].abs().mean())
        h_imp.append(hv); m_imp.append(mv)
        shap_rows.append({"Crop": CROP_LABELS[crop], "Feature": feat,
                          "Mean_abs_SHAP_H": hv, "Mean_abs_SHAP_M": mv})

    h_rank = pd.Series(h_imp, index=BASE_FEATURES + ["SPEI3"]).rank(ascending=False, method="min")
    m_rank = pd.Series(m_imp, index=BASE_FEATURES + ["SPEI3"]).rank(ascending=False, method="min")
    for row in shap_rows[-7:]:
        feat = row["Feature"]
        row["Rank_H"] = int(h_rank[feat]); row["Rank_M"] = int(m_rank[feat])

    summary_rows.append({
        "Crop": CROP_LABELS[crop],
        "SPEI_H_M_r": spei_corr,
        "SHAP_importance_r": pearsonr(h_imp, m_imp).statistic,
        "SHAP_rank_rho": spearmanr(h_imp, m_imp).statistic,
        "SPEI3_rank_H": int(h_rank["SPEI3"]),
        "SPEI3_rank_M": int(m_rank["SPEI3"]),
    })

    hh = filter_hist(pd.read_csv(RF_H_DIR / crop / f"RF_{crop}_historical.csv"))
    mm = filter_hist(pd.read_csv(RF_M_DIR / crop / f"RF_{crop}_historical.csv"))

    for period, code in PERIOD_FILES.items():
        fh = pd.read_csv(RF_H_DIR / crop / f"RF_{crop}_{code}.csv")
        fm = pd.read_csv(RF_M_DIR / crop / f"RF_{crop}_{code}.csv")
        for scenario in SCENARIOS:
            sh = fh[fh["scenario"] == scenario]
            sm = fm[fm["scenario"] == scenario]
            common = sorted(set(hh["model"]) & set(mm["model"]) & set(sh["model"]) & set(sm["model"]))
            for model in common:
                # Exclude a model when the Makkink SPEI field itself is absent, even if RF median
                # imputation produced a numerical prediction.
                mg = sm[sm["model"] == model]
                if "SPEI3_Makkink" not in mg or mg["SPEI3_Makkink"].notna().sum() == 0:
                    continue
                dh = sh.loc[sh["model"] == model, "pred_anomaly"].mean() - hh.loc[hh["model"] == model, "pred_anomaly"].mean()
                dm = sm.loc[sm["model"] == model, "pred_anomaly"].mean() - mm.loc[mm["model"] == model, "pred_anomaly"].mean()
                if np.isfinite(dh) and np.isfinite(dm):
                    model_detail_rows.append({"Crop": CROP_LABELS[crop], "Period": period,
                                              "Scenario": SCEN_LABEL[scenario], "Model": model,
                                              "Delta_H": dh, "Delta_M": dm})

summary = pd.DataFrame(summary_rows)
hist_metrics = pd.DataFrame(hist_metric_rows)
shap_table = pd.DataFrame(shap_rows)[["Crop", "Feature", "Mean_abs_SHAP_H", "Rank_H", "Mean_abs_SHAP_M", "Rank_M"]]
detail = pd.DataFrame(model_detail_rows)

agg_rows = []
for keys, g in detail.groupby(["Crop", "Period", "Scenario"], sort=False):
    dh, dm = g["Delta_H"].to_numpy(), g["Delta_M"].to_numpy()
    agg_rows.append({
        "Crop": keys[0], "Period": keys[1], "Scenario": keys[2], "N_valid_models": len(g),
        "Mean_delta_H": dh.mean(), "Median_delta_H": np.median(dh),
        "Mean_delta_M": dm.mean(), "Median_delta_M": np.median(dm),
        "r_models": pearsonr(dh, dm).statistic if len(g) > 1 else np.nan,
        "Same_sign_n": int(np.sum(np.sign(dh) == np.sign(dm))),
        "Same_sign_fraction": float(np.mean(np.sign(dh) == np.sign(dm))),
        "N_negative_H": int(np.sum(dh < 0)), "N_negative_M": int(np.sum(dm < 0)),
    })
future_summary = pd.DataFrame(agg_rows)

summary.to_csv(OUT_DIR / "Table_S1_PET_summary.csv", index=False)
hist_metrics.to_csv(OUT_DIR / "Table_S1a_PET_historical_metrics.csv", index=False)
shap_table.to_csv(OUT_DIR / "Table_S1b_PET_SHAP.csv", index=False)
future_summary.to_csv(OUT_DIR / "Table_S2_PET_future_sensitivity_valid_models.csv", index=False)
detail.to_csv(OUT_DIR / "Table_S2b_PET_model_detail.csv", index=False)

print(summary.to_string(index=False))
print("\nFuture PET sensitivity")
print(future_summary.to_string(index=False))
