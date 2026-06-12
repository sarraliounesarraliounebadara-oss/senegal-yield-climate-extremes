import xarray as xr
import pandas as pd
import numpy as np
import os
import glob

# =====================================
# PATHS
# =====================================
MASK_DIR = (
    "/media/sarr/01DC5DE7BBEA1120/"
    "Donnees/mirca_masks_nc"
)

YIELD_DIR = (
    "/media/sarr/01DC5DE9D15E8CF0/"
    "dossier/yield_data"
)

OUT_DIR = (
    "/media/sarr/01DC5DE9D15E8CF0/"
    "results_agro/results_agro2"
)

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================
# CROPS
# =====================================
crops = [
    "maize",
    "millet",
    "groundnut",
    "sorghum"
]

# =====================================
# LOOP CROPS
# =====================================
for crop in crops:

    print("\n========================")
    print(f"CROP: {crop}")
    print("========================")

    # =====================================
    # READ YIELD FILE
    # =====================================
    yield_file = (
        f"{YIELD_DIR}/{crop}_yield.txt"
    )

    if not os.path.exists(yield_file):
        print(f"⚠️ Fichier introuvable, passé : {yield_file}")
        continue

    yield_df = pd.read_csv(
        yield_file,
        sep="\t",
        engine="python"
    )

    # =====================================
    # CLEAN COLUMNS
    # =====================================
    yield_df.columns = [
        c.strip().lower()
        for c in yield_df.columns
    ]

    # =====================================
    # CLEAN DEPARTMENT
    # =====================================
    yield_df["department"] = (
        yield_df["department"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # =====================================
    # KEEP COLUMNS
    # =====================================
    yield_df = yield_df[[
        "department",
        "year",
        "yield_anomaly"
    ]]

    # =====================================
    # MASK FILES
    # =====================================
    files = sorted(
        glob.glob(
            f"{MASK_DIR}/{crop}_*_mask.nc"
        )
    )

    all_rows = []

    # =====================================
    # LOOP MASK FILES
    # =====================================
    for f in files:

        fname = os.path.basename(f)
        print(f"Processing: {fname}")

        # =====================================
        # DEPARTMENT
        # =====================================
        department = (
            fname
            .replace(f"{crop}_", "")
            .replace("_mask.nc", "")
            .strip()
            .lower()
        )
        # Harmonisation noms
        department = department.replace("nioro_du_rip", "nioro")

        # =====================================
        # OPEN NETCDF
        # =====================================
        with xr.open_dataset(f) as ds:
            crop_area = ds["crop_area"]
            weight = ds["weight"]

            # =====================================
            # TO DATAFRAME
            # =====================================
            df = crop_area.to_dataframe().reset_index()
            df_weight = weight.to_dataframe().reset_index()

        # =====================================
        # MERGE WEIGHT
        # =====================================
        df = df.merge(
            df_weight,
            on=["lat", "lon"]
        )

        # =====================================
        # REMOVE NAN
        # =====================================
        df = df.dropna(
            subset=["crop_area"]
        )

        # =====================================
        # GRID IDS
        # =====================================
        df["grid"] = [
            f"{department}_{i+1}"
            for i in range(len(df))
        ]

        # =====================================
        # ADD METADATA
        # =====================================
        df["crop"] = crop
        df["department"] = department

        # =====================================
        # YIELD DATA FOR DEPARTMENT
        # =====================================
        dep_yield = yield_df[
            yield_df["department"] == department
        ].copy()

        # =====================================
        # REPLICATE YEARS FOR EACH GRID (OPTIMIZED)
        # =====================================
        dep_final = df.merge(dep_yield, on="department")

        all_rows.append(dep_final)

    # =====================================
    # CONCAT ALL DEPARTMENTS
    # =====================================
    if all_rows:
        final_df = pd.concat(
            all_rows,
            ignore_index=True
        )

        # =====================================
        # SORT
        # =====================================
        final_df = final_df.sort_values(
            by=[
                "department",
                "grid",
                "year"
            ]
        )

        # =====================================
        # COLUMN ORDER
        # =====================================
        final_df = final_df[[
            "crop",
            "department",
            "grid",
            "year",
            "lon",
            "lat",
            "crop_area",
            "weight",
            "yield_anomaly"
        ]]

        # =====================================
        # SAVE
        # =====================================
        outfile = (
            f"{OUT_DIR}/{crop}_grids.csv"
        )

        final_df.to_csv(
            outfile,
            index=False
        )

        print(f"Saved: {outfile}")

print("\nDONE")
