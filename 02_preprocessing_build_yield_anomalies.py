# =========================================
# BUILD YIELD ANOMALY (LOESS DETRENDING)
# =========================================

import pandas as pd
import numpy as np
import os
import unicodedata
from statsmodels.nonparametric.smoothers_lowess import lowess

# ==========================
# PATH
# ==========================
BASE = "/media/sarr/01DC5DE9D15E8CF0/dossier/yield_data"

files = [
    "groundnut_yield_clean.xlsx",
    "maize_yield_clean.xlsx",
    "millet_yield_clean.xlsx",
    "sorghum_yield_clean.xlsx"
]

# ==========================
# CLEAN FUNCTION
# ==========================
def clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("utf-8")
    return x.replace(" ", "_")

# ==========================
# LOOP OVER CROPS
# ==========================
for f in files:

    path = os.path.join(BASE, f)
    if not os.path.exists(path):
        continue

    print("\nProcessing:", f)

    df = pd.read_excel(path)

    # ==========================
    # CLEAN
    # ==========================
    df.columns = [c.lower() for c in df.columns]
    df["department"] = df["department"].apply(clean_text)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

    df = df.sort_values(["department", "year"])

    # ==========================
    # INIT
    # ==========================
    df["yield_trend"] = np.nan
    df["yield_anomaly"] = np.nan

    # ==========================
    # LOESS PER DEPARTMENT
    # ==========================
    for dep, group in df.groupby("department"):

        g = group.dropna(subset=["yield", "year"])

        if len(g) >= 6:

            loess_fit = lowess(
                g["yield"],
                g["year"],
                frac=0.4,
                return_sorted=True
            )

            trend_all = np.interp(
                group["year"],
                loess_fit[:, 0],
                loess_fit[:, 1]
            )

        else:
            trend_all = np.full(len(group), np.nan)

        df.loc[group.index, "yield_trend"] = trend_all

        # ==========================
        # ANOMALY (%)
        # ==========================
        df.loc[group.index, "yield_anomaly"] = np.where(
            trend_all != 0,
            (group["yield"] - trend_all) * 100 / trend_all,
            np.nan
        )

    # ==========================
    # SAVE
    # ==========================
    out = path.replace("_clean.xlsx", ".txt")
    df.to_csv(out, sep="\t", index=False)

    print("✅ anomaly file:", out)

print("\n🎯 ANOMALY DONE")
