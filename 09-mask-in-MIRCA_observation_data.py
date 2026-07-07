#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import numpy as np
import os
import gc

from scipy.spatial import cKDTree

# =====================================================
# PATHS
# =====================================================

BASE = "/media/sarr/01DC5DE9D15E8CF0"

EXTREMES_FILE = (
    f"{BASE}/departments/climate_data/"
    f"extremes/observations/"
    f"observations_extremes.nc"
)

GRID_DIR = (
    f"{BASE}/results_agro/results_agro2"
)

OUT_DIR = (
    f"{BASE}/departments/climate_data/"
    f"extremes/crop_tables"
)

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# CROPS
# =====================================================

crops = [
    "maize",
    "millet",
    "groundnut",
    "sorghum"
]

# =====================================================
# CLIMATE VARIABLES
# =====================================================

climate_vars = [

    "CDD",
    "CWD",
    "PRtot",
    "R95pTOT",
    "RX1day",
    "SPEI3",
    "WSDI",
    "TXx",
    "TNx",
    "DTR",
    "TX90p",
    "TN90p",
    "TX35",
    "RX5day",
    "SDII",
    "R1mm",
    "R10mm",
    "R20mm"

]

# =====================================================
# LOAD EXTREMES
# =====================================================

print("\nLoading extremes...")

ds = xr.open_dataset(
    EXTREMES_FILE
)

# =====================================================
# DETECT COORDS
# =====================================================

lon_name = None
lat_name = None

for c in ds.coords:

    cname = c.lower()

    if "lon" in cname:
        lon_name = c

    if "lat" in cname:
        lat_name = c

print("\nCoordinates detected:")
print("lon =", lon_name)
print("lat =", lat_name)

# =====================================================
# KEEP ONLY NECESSARY VARIABLES
# =====================================================

existing_vars = [

    v for v in climate_vars
    if v in ds.data_vars

]

print("\nVariables:")
print(existing_vars)

# =====================================================
# EXTRACT ONLY REQUIRED VARIABLES
# =====================================================

df_ext = (

    ds[existing_vars]

    .to_dataframe()

    .reset_index()

)

# =====================================================
# CLOSE DATASET
# =====================================================

ds.close()

gc.collect()

# =====================================================
# RENAME COORDS
# =====================================================

df_ext = df_ext.rename(columns={

    lon_name: "lon",
    lat_name: "lat"

})

# =====================================================
# YEAR
# =====================================================

df_ext["year"] = pd.to_datetime(
    df_ext["time"]
).dt.year.astype("int32")

# =====================================================
# MEMORY OPT
# =====================================================

float_cols = df_ext.select_dtypes(
    include=["float64"]
).columns

for col in float_cols:

    df_ext[col] = (
        df_ext[col]
        .astype("float32")
    )

# =====================================================
# UNIQUE PIXELS
# =====================================================

pixels = df_ext[
    ["lon", "lat"]
].drop_duplicates()

# =====================================================
# KD TREE
# =====================================================

tree = cKDTree(
    pixels[["lon", "lat"]].values
)

# =====================================================
# LOOP CROPS
# =====================================================

for crop in crops:

    print("\n=================================")
    print(crop)
    print("=================================")

    # =================================================
    # LOAD GRID
    # =================================================

    grid_file = (
        f"{GRID_DIR}/{crop}_grids.csv"
    )

    if not os.path.exists(grid_file):

        print("Missing:")
        print(grid_file)

        continue

    df_grid = pd.read_csv(
        grid_file
    )

    # =================================================
    # REMOVE MONTH
    # =================================================

    if "month" in df_grid.columns:

        df_grid.drop(
            columns=["month"],
            inplace=True
        )

    # =================================================
    # MEMORY OPT
    # =================================================

    for col in ["lon", "lat"]:

        df_grid[col] = (
            df_grid[col]
            .astype("float32")
        )

    # =================================================
    # FIND NEAREST PIXELS
    # =================================================

    dist, idx = tree.query(

        df_grid[
            ["lon", "lat"]
        ].values,

        k=1

    )

    print("Max distance :", dist.max())
    print("Mean distance:", dist.mean())

    if dist.max() > 0.05:
        print("WARNING")

    nearest = pixels.iloc[
        idx
    ].reset_index(drop=True)

    df_grid["lon_match"] = (
        nearest["lon"].values
    )

    df_grid["lat_match"] = (
        nearest["lat"].values
    )

    # =================================================
    # MERGE
    # =================================================

    df_merge = pd.merge(

        df_grid,
        df_ext,

        left_on=[
            "lon_match",
            "lat_match",
            "year"
        ],

        right_on=[
            "lon",
            "lat",
            "year"
        ],

        how="left"

    )

    # =================================================
    # CLEAN
    # =================================================

    drop_cols = [

        "lon_y",
        "lat_y",
        "lon_match",
        "lat_match",
        "time"

    ]

    for col in drop_cols:

        if col in df_merge.columns:

            df_merge.drop(
                columns=col,
                inplace=True
            )

    df_merge = df_merge.rename(columns={

        "lon_x": "lon",
        "lat_x": "lat"

    })

    # =================================================
    # VARIABLES TO AVERAGE
    # =====================================================

    mean_vars = []

    for col in climate_vars:

        if col in df_merge.columns:
            mean_vars.append(col)

    yield_vars = [

        "yield_anomaly",
        "yield",
        "mean_yield",
        "yield_change"

    ]

    for col in yield_vars:

        if col in df_merge.columns:
            mean_vars.append(col)

    # =================================================
    # GROUP COLS
    # =====================================================

    group_cols = []

    if "department" in df_merge.columns:
        group_cols.append("department")

    if "year" in df_merge.columns:
        group_cols.append("year")

    # =================================================
    # WEIGHTED MEAN
    # =====================================================

    print("\nComputing department averages...")

    if "weight" in df_merge.columns:

        print("Using weighted means...")

        rows = []

        grouped = df_merge.groupby(
            group_cols
        )

        for keys, g in grouped:

            row = {}

            if len(group_cols) == 2:

                row["department"] = keys[0]
                row["year"] = keys[1]

            else:

                row[group_cols[0]] = keys

            for var in mean_vars:

                valid = g[
                    [var, "weight"]
                ].dropna()

                if len(valid) > 0:

                    row[var] = np.average(

                        valid[var],

                        weights=valid["weight"]

                    )

                else:

                    row[var] = np.nan

            rows.append(row)

        df_mean = pd.DataFrame(rows)

    else:

        print("Using simple means...")

        df_mean = (

            df_merge

            .groupby(group_cols)[mean_vars]

            .mean()

            .reset_index()

        )

    # =================================================
    # SAVE
    # =====================================================

    out_csv = (
        f"{OUT_DIR}/{crop}_extremes.csv"
    )

    df_mean.to_csv(
        out_csv,
        index=False
    )

    print("\nSaved:")
    print(out_csv)

    del df_grid
    del df_merge
    del df_mean

    gc.collect()

# =====================================================
# CLEAN MEMORY
# =====================================================

del df_ext
del pixels

gc.collect()

# =====================================================
# END
# =====================================================

print("\n=================================")
print("DONE")
print("=================================")
