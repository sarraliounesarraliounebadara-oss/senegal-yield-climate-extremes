# ==========================================
# PREPARE YIELD DATA (DAPSA)
# ==========================================

import pandas as pd
import os

# ==========================
# PATHS
# ==========================
DAPSA_FILE = "/media/sarr/01DC5DE9D15E8CF0/dossier/DAPSA.xlsx"
OUTDIR = "/media/sarr/01DC5DE9D15E8CF0/dossier/yield_data"

os.makedirs(OUTDIR, exist_ok=True)

# ==========================
# LOAD DATA
# ==========================
df = pd.read_excel(
    DAPSA_FILE,
    sheet_name="80_2013_selon80",
    header=1
)

# ==========================
# CLEANING
# ==========================
df["departement"] = df["departement"].str.strip()
df["culture"] = df["culture"].str.strip()

# ==========================
# CROP MAPPING
# ==========================
crops = {
    "ARACHIDE": "groundnut",
    "MAIS": "maize",
    "MIL": "millet",
    "SORGHO": "sorghum"
}

# ==========================
# STUDY AREA (24 DEPARTMENTS)
# ==========================
departments = sorted([
    "Bakel","Bambey","Bignona","Diourbel","Fatick",
    "Foundiougne","Gossas","Kaffrine","Kaolack","Kebemer",
    "Kedougou","Kolda","Linguere","Louga","Mbacke",
    "Mbour","Nioro","Oussouye","Sedhiou",
    "Tambacounda","Thies","Tivaouane","Velingara","Ziguinchor"
])

years = list(range(1984, 2014))

# ==========================
# FILTER PERIOD
# ==========================
df = df[df["annee"].between(1984,2013)]

# ==========================
# PROCESS EACH CROP
# ==========================
for crop_fr, crop_en in crops.items():

    data = df[df["culture"] == crop_fr]

    # pivot table (department × year)
    pivot = data.pivot_table(
        index="departement",
        columns="annee",
        values="Rdt",
        aggfunc="first"
    )

    # ensure full structure
    pivot = pivot.reindex(index=departments, columns=years)

    # convert to long format
    out = pivot.stack(dropna=False).reset_index()
    out.columns = ["department", "year", "yield"]

    # save
    outfile = f"{OUTDIR}/{crop_en}_yield.xlsx"
    out.to_excel(outfile, index=False)

    print(f"{crop_en}: {len(out)} rows -> {outfile}")
