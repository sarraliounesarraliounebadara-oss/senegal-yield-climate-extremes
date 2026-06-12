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

EXTREMES_DIR = (
    f"{BASE}/departments/climate_data/extremes"
)

GRID_DIR = (
    f"{BASE}/results_agro/results_agro2"
)

OUT_DIR = (
    f"{BASE}/departments/climate_data/extremes/cmip6_crop_tables"
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
# SCENARIOS
# =====================================================

scenarios = [
    "ssp245",
    "ssp585"
]

# =====================================================
# VARIABLES
# =====================================================

vars_keep = [

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
# SAFE YEAR EXTRACTION
# =====================================================

def extract_year(time_col):

    try:

        return pd.to_datetime(
            time_col
        ).dt.year.astype("int32")

    except Exception:

        return pd.Series(
            [t.year for t in time_col],
            dtype="int32"
        )

# =====================================================
# MEMORY OPT
# =====================================================

def optimize_dataframe(df):

    float_cols = df.select_dtypes(
        include=["float64"]
    ).columns

    for col in float_cols:

        df[col] = (
            df[col]
            .astype("float32")
        )

    int_cols = df.select_dtypes(
        include=["int64"]
    ).columns

    for col in int_cols:

        df[col] = (
            df[col]
            .astype("int32")
        )

    return df

# =====================================================
# PROCESS FUNCTION
# =====================================================

def process_extremes_file(
    ncfile,
    df_grid,
    model,
    scenario=None,
    future=False
):

    print(f"\nProcessing: {os.path.basename(ncfile)}")

    # =================================================
    # OPEN DATASET
    # =================================================

    ds = xr.open_dataset(
        ncfile
    )

    # =================================================
    # DETECT COORDS
    # =================================================

    lon_name = None
    lat_name = None

    for c in ds.coords:

        cname = c.lower()

        if "lon" in cname:
            lon_name = c

        if "lat" in cname:
            lat_name = c

    if lon_name is None or lat_name is None:

        print("Could not detect lon/lat.")

        ds.close()

        return None

    # =================================================
    # KEEP VARIABLES
    # =================================================

    existing_vars = [

        v for v in vars_keep
        if v in ds.data_vars

    ]

    print("\nVariables found:")
    print(existing_vars)

    if len(existing_vars) == 0:

        print("No climate variables.")

        ds.close()

        return None

    # =================================================
    # DATAFRAME
    # =================================================

    df_ext = (

        ds[existing_vars]

        .to_dataframe()

        .reset_index()

    )

    ds.close()

    # =================================================
    # RENAME COORDS
    # =================================================

    df_ext.rename(

        columns={
            lon_name: "lon",
            lat_name: "lat"
        },

        inplace=True

    )

    # =================================================
    # YEAR
    # =================================================

    df_ext["year"] = extract_year(
        df_ext["time"]
    )

    print(
        "\nClimate years:",
        df_ext["year"].min(),
        "-",
        df_ext["year"].max()
    )

    # =================================================
    # MEMORY OPT
    # =================================================

    df_ext = optimize_dataframe(
        df_ext
    )

    # =================================================
    # PIXELS
    # =================================================

    pixels = df_ext[
        ["lon", "lat"]
    ].drop_duplicates()

    # =================================================
    # KD TREE
    # =================================================

    tree = cKDTree(
        pixels[["lon", "lat"]].values
    )

    # =================================================
    # NEAREST PIXELS
    # =================================================

    dist, idx = tree.query(

        df_grid[
            ["lon", "lat"]
        ].values,

        k=1

    )

    print(
        "Max distance:",
        dist.max()
    )

    nearest = pixels.iloc[
        idx
    ].reset_index(drop=True)

    # =================================================
    # GRID COPY
    # =================================================

    df_tmp = df_grid.copy()

    df_tmp["lon_match"] = (
        nearest["lon"].values
    )

    df_tmp["lat_match"] = (
        nearest["lat"].values
    )

    # =================================================
    # MERGE
    # =================================================

    # HISTORICAL:
    # merge on year

    if future is False:

        print("\nHistorical merge")

        print(
            "Grid years:",
            df_tmp["year"].min(),
            "-",
            df_tmp["year"].max()
        )

        df_merge = pd.merge(

            df_tmp,
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

    # FUTURE:
    # DO NOT MERGE ON YEAR
    # because MIRCA years are historical
    # while CMIP6 years are future

    else:

        print("\nFuture merge")

        print(
            "Grid years:",
            df_tmp["year"].min(),
            "-",
            df_tmp["year"].max()
        )

        print(
            "Climate years:",
            df_ext["year"].min(),
            "-",
            df_ext["year"].max()
        )

        # remove grid year
        if "year" in df_tmp.columns:

            df_tmp = df_tmp.drop(
                columns=["year"]
            )

        df_merge = pd.merge(

            df_tmp,
            df_ext,

            left_on=[
                "lon_match",
                "lat_match"
            ],

            right_on=[
                "lon",
                "lat"
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

    df_merge.rename(

        columns={
            "lon_x": "lon",
            "lat_x": "lat"
        },

        inplace=True

    )

    # =================================================
    # REMOVE MONTH
    # =================================================

    if "month" in df_merge.columns:

        df_merge.drop(
            columns=["month"],
            inplace=True
        )

    # =================================================
    # DEPARTMENT AVERAGES
    # =====================================================

    mean_vars = []

    for col in vars_keep:

        if col in df_merge.columns:
            mean_vars.append(col)

    group_cols = []

    if "department" in df_merge.columns:
        group_cols.append("department")

    if "year" in df_merge.columns:
        group_cols.append("year")

    # =================================================
    # WEIGHTED MEAN
    # =====================================================

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

        row["model"] = model

        if scenario is not None:
            row["scenario"] = scenario

        rows.append(row)

    df_mean = pd.DataFrame(rows)

    # =================================================
    # NaN CHECK
    # =====================================================

    print("\nNaN ratio:")

    print(
        df_mean[mean_vars]
        .isna()
        .mean()
    )

    # =================================================
    # CLEAN MEMORY
    # =================================================

    del df_ext
    del pixels
    del nearest
    del df_tmp
    del df_merge

    gc.collect()

    return df_mean

# =====================================================
# LOOP CROPS
# =====================================================

for crop in crops:

    print("\n====================================")
    print(crop)
    print("====================================")

    # =================================================
    # LOAD GRID
    # =====================================================

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
    # =====================================================

    if "month" in df_grid.columns:

        df_grid.drop(
            columns=["month"],
            inplace=True
        )

    # =================================================
    # MEMORY OPT
    # =====================================================

    df_grid = optimize_dataframe(
        df_grid
    )

    # =================================================
    # HISTORICAL
    # =====================================================

    hist_all = []

    hist_dir = (
        f"{EXTREMES_DIR}/historical"
    )

    hist_files = sorted([

        f for f in os.listdir(hist_dir)

        if "historical_extremes" in f

    ])

    for file in hist_files:

        model = (
            file.replace(
                "_historical_extremes.nc",
                ""
            )
        )

        ncfile = (
            f"{hist_dir}/{file}"
        )

        df_model = process_extremes_file(

            ncfile=ncfile,
            df_grid=df_grid,
            model=model,
            future=False

        )

        if df_model is not None:

            hist_all.append(
                df_model
            )

    # =================================================
    # SAVE HIST
    # =====================================================

    if len(hist_all) > 0:

        hist_df = pd.concat(
            hist_all,
            ignore_index=True
        )

        hist_out = (
            f"{OUT_DIR}/"
            f"{crop}_historical_extremes.csv"
        )

        hist_df.to_csv(
            hist_out,
            index=False
        )

        print("\nSaved:")
        print(hist_out)

        del hist_df

        gc.collect()

    # =================================================
    # NEAR FUTURE
    # =====================================================

    near_all = []

    for scenario in scenarios:

        scen_dir = (
            f"{EXTREMES_DIR}/{scenario}"
        )

        files = sorted([

            f for f in os.listdir(
                scen_dir
            )

            if "_near_extremes.nc" in f

        ])

        for file in files:

            model = file.split(
                f"_{scenario}_near"
            )[0]

            ncfile = (
                f"{scen_dir}/{file}"
            )

            df_model = process_extremes_file(

                ncfile=ncfile,
                df_grid=df_grid,
                model=model,
                scenario=scenario,
                future=True

            )

            if df_model is not None:

                near_all.append(
                    df_model
                )

    # =================================================
    # SAVE NEAR
    # =====================================================

    if len(near_all) > 0:

        near_df = pd.concat(
            near_all,
            ignore_index=True
        )

        near_out = (
            f"{OUT_DIR}/"
            f"{crop}_near_future_extremes.csv"
        )

        near_df.to_csv(
            near_out,
            index=False
        )

        print("\nSaved:")
        print(near_out)

        del near_df

        gc.collect()

    # =================================================
    # FAR FUTURE
    # =====================================================

    far_all = []

    for scenario in scenarios:

        scen_dir = (
            f"{EXTREMES_DIR}/{scenario}"
        )

        files = sorted([

            f for f in os.listdir(
                scen_dir
            )

            if "_far_extremes.nc" in f

        ])

        for file in files:

            model = file.split(
                f"_{scenario}_far"
            )[0]

            ncfile = (
                f"{scen_dir}/{file}"
            )

            df_model = process_extremes_file(

                ncfile=ncfile,
                df_grid=df_grid,
                model=model,
                scenario=scenario,
                future=True

            )

            if df_model is not None:

                far_all.append(
                    df_model
                )

    # =================================================
    # SAVE FAR
    # =====================================================

    if len(far_all) > 0:

        far_df = pd.concat(
            far_all,
            ignore_index=True
        )

        far_out = (
            f"{OUT_DIR}/"
            f"{crop}_far_future_extremes.csv"
        )

        far_df.to_csv(
            far_out,
            index=False
        )

        print("\nSaved:")
        print(far_out)

        del far_df

        gc.collect()

# =====================================================
# END
# =====================================================

print("\n====================================")
print("DONE")
print("====================================")
