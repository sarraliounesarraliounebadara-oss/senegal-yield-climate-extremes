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

BASE = "/mnt/disque"

EXTREMES_FILE = (
    f"{BASE}/departments/climate_data/"
    f"extremes/observations/"
    f"observations_extremes.nc"
)

# Nouveau SPEI3-Makkink observations
SPEI3_MAKKINK_FILE = (
    f"{BASE}/departments/climate_data/"
    f"anacim_data/"
    f"observations_spei3_makkink_1984-2013.nc"
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

# L'ancien SPEI3 contenu dans observations_extremes.nc
# sera renommé SPEI3_Hargreaves.
# Le nouveau fichier sera ajouté comme SPEI3_Makkink.

climate_vars = [

    "CDD",
    "CWD",
    "PRtot",
    "R95pTOT",
    "RX1day",

    "SPEI3_Hargreaves",
    "SPEI3_Makkink",

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
# HELPER: DETECT COORDINATES
# =====================================================

def detect_lon_lat(ds):

    lon_name = None
    lat_name = None

    # Cherche d'abord dans les coords
    for c in ds.coords:

        cname = c.lower()

        if "lon" in cname:
            lon_name = c

        if "lat" in cname:
            lat_name = c

    # Si nécessaire, cherche aussi dans les dimensions
    if lon_name is None:
        for d in ds.dims:
            if "lon" in d.lower():
                lon_name = d
                break

    if lat_name is None:
        for d in ds.dims:
            if "lat" in d.lower():
                lat_name = d
                break

    if lon_name is None or lat_name is None:
        raise ValueError(
            f"Impossible de détecter lon/lat. "
            f"coords={list(ds.coords)}, dims={list(ds.dims)}"
        )

    return lon_name, lat_name


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

lon_name, lat_name = detect_lon_lat(ds)

print("\nCoordinates detected in extremes:")
print("lon =", lon_name)
print("lat =", lat_name)

# =====================================================
# RENAME ORIGINAL SPEI3
# =====================================================

# Dans l'ancien fichier observations_extremes.nc,
# SPEI3 correspond au SPEI3-Hargreaves.

if "SPEI3" in ds.data_vars:

    print(
        "\nRenaming original SPEI3 -> SPEI3_Hargreaves"
    )

    ds = ds.rename({
        "SPEI3": "SPEI3_Hargreaves"
    })

elif "SPEI3_Hargreaves" in ds.data_vars:

    print(
        "\nSPEI3_Hargreaves already present."
    )

else:

    print(
        "\nWARNING: neither SPEI3 nor "
        "SPEI3_Hargreaves found in extremes file."
    )

# =====================================================
# KEEP ONLY NECESSARY VARIABLES
# =====================================================

# SPEI3_Makkink n'est pas encore dans ds :
# il sera ajouté ensuite par merge pandas.

extreme_vars = [
    v for v in climate_vars
    if v != "SPEI3_Makkink"
    and v in ds.data_vars
]

print("\nVariables from observations_extremes.nc:")
print(extreme_vars)

# =====================================================
# EXTRACT EXTREMES TO DATAFRAME
# =====================================================

df_ext = (

    ds[extreme_vars]

    .to_dataframe()

    .reset_index()

)

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
# COORDINATE NORMALIZATION
# =====================================================

# Même grille attendue, mais on arrondit légèrement
# pour éviter des différences numériques infinitésimales
# lors du merge avec SPEI3-Makkink.

df_ext["lon"] = (
    df_ext["lon"]
    .astype("float64")
    .round(6)
)

df_ext["lat"] = (
    df_ext["lat"]
    .astype("float64")
    .round(6)
)

# =====================================================
# MEMORY OPT
# =====================================================

float_cols = df_ext.select_dtypes(
    include=["float64"]
).columns

for col in float_cols:

    if col not in ["lon", "lat"]:

        df_ext[col] = (
            df_ext[col]
            .astype("float32")
        )

# =====================================================
# LOAD NEW SPEI3-MAKKINK
# =====================================================

print("\nLoading SPEI3-Makkink...")

if not os.path.exists(
    SPEI3_MAKKINK_FILE
):

    raise FileNotFoundError(
        SPEI3_MAKKINK_FILE
    )

ds_m = xr.open_dataset(
    SPEI3_MAKKINK_FILE
)

lon_m, lat_m = detect_lon_lat(
    ds_m
)

print("\nCoordinates detected in SPEI3-Makkink:")
print("lon =", lon_m)
print("lat =", lat_m)

if "SPEI3" in ds_m.data_vars:

    spei3_m_name = "SPEI3"

elif "SPEI3_Makkink" in ds_m.data_vars:

    spei3_m_name = "SPEI3_Makkink"

else:

    raise KeyError(
        "No SPEI3 variable found in "
        f"{SPEI3_MAKKINK_FILE}. "
        f"Variables: {list(ds_m.data_vars)}"
    )

# =====================================================
# SPEI3-MAKKINK -> DATAFRAME
# =====================================================

df_m = (

    ds_m[
        [spei3_m_name]
    ]

    .to_dataframe()

    .reset_index()

)

ds_m.close()

gc.collect()

df_m = df_m.rename(columns={

    lon_m: "lon",
    lat_m: "lat",
    spei3_m_name: "SPEI3_Makkink"

})

df_m["year"] = pd.to_datetime(
    df_m["time"]
).dt.year.astype("int32")

df_m["lon"] = (
    df_m["lon"]
    .astype("float64")
    .round(6)
)

df_m["lat"] = (
    df_m["lat"]
    .astype("float64")
    .round(6)
)

df_m["SPEI3_Makkink"] = (
    df_m["SPEI3_Makkink"]
    .astype("float32")
)

# Une seule valeur par pixel et année
df_m = df_m[
    [
        "lon",
        "lat",
        "year",
        "SPEI3_Makkink"
    ]
]

# =====================================================
# MERGE EXTREMES + SPEI3-MAKKINK
# =====================================================

print(
    "\nMerging SPEI3-Makkink with "
    "observational extremes..."
)

n_before = len(df_ext)

df_ext = pd.merge(

    df_ext,
    df_m,

    on=[
        "lon",
        "lat",
        "year"
    ],

    how="left",
    validate="many_to_one"

)

if len(df_ext) != n_before:

    raise RuntimeError(
        "Unexpected change in number of rows "
        "after SPEI3-Makkink merge."
    )

n_valid_m = (
    df_ext["SPEI3_Makkink"]
    .notna()
    .sum()
)

print(
    "Valid SPEI3_Makkink values after merge:",
    n_valid_m,
    "/",
    len(df_ext)
)

# Diagnostic par année
years_m = sorted(
    df_ext.loc[
        df_ext["SPEI3_Makkink"].notna(),
        "year"
    ].unique()
)

if years_m:

    print(
        "SPEI3-Makkink years:",
        years_m[0],
        "->",
        years_m[-1]
    )

# Plus besoin du tableau séparé
del df_m
gc.collect()

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
    pixels[
        ["lon", "lat"]
    ].values
)

# =====================================================
# LOOP CROPS
# =====================================================

for crop in crops:

    print("\n=================================")
    print(crop)
    print("=================================")

    # =================================================
    # LOAD MIRCA CROP GRID
    # =================================================

    grid_file = (
        f"{GRID_DIR}/{crop}_grids.csv"
    )

    if not os.path.exists(
        grid_file
    ):

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
    # COORDINATE NORMALIZATION
    # =================================================

    for col in [
        "lon",
        "lat"
    ]:

        df_grid[col] = (
            df_grid[col]
            .astype("float64")
            .round(6)
        )

    # =================================================
    # FIND NEAREST CLIMATE PIXELS
    # =================================================

    dist, idx = tree.query(

        df_grid[
            ["lon", "lat"]
        ].values,

        k=1

    )

    print(
        "Max distance :",
        dist.max()
    )

    print(
        "Mean distance:",
        dist.mean()
    )

    if dist.max() > 0.05:
        print("WARNING")

    nearest = pixels.iloc[
        idx
    ].reset_index(
        drop=True
    )

    df_grid["lon_match"] = (
        nearest["lon"].values
    )

    df_grid["lat_match"] = (
        nearest["lat"].values
    )

    # =================================================
    # MERGE MIRCA GRID + CLIMATE
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
    # =================================================

    mean_vars = []

    for col in climate_vars:

        if col in df_merge.columns:

            mean_vars.append(
                col
            )

    yield_vars = [

        "yield_anomaly",
        "yield",
        "mean_yield",
        "yield_change"

    ]

    for col in yield_vars:

        if col in df_merge.columns:

            mean_vars.append(
                col
            )

    print(
        "\nVariables averaged:"
    )

    print(
        mean_vars
    )

    # =================================================
    # GROUP COLS
    # =================================================

    group_cols = []

    if "department" in df_merge.columns:

        group_cols.append(
            "department"
        )

    if "year" in df_merge.columns:

        group_cols.append(
            "year"
        )

    # =================================================
    # WEIGHTED MEAN
    # =================================================

    print(
        "\nComputing department averages..."
    )

    if "weight" in df_merge.columns:

        print(
            "Using MIRCA weighted means..."
        )

        rows = []

        grouped = df_merge.groupby(
            group_cols
        )

        for keys, g in grouped:

            row = {}

            if len(
                group_cols
            ) == 2:

                row[
                    "department"
                ] = keys[0]

                row[
                    "year"
                ] = keys[1]

            else:

                row[
                    group_cols[0]
                ] = keys

            for var in mean_vars:

                valid = g[
                    [
                        var,
                        "weight"
                    ]
                ].dropna()

                if len(
                    valid
                ) > 0:

                    row[var] = np.average(

                        valid[var],

                        weights=valid[
                            "weight"
                        ]

                    )

                else:

                    row[var] = np.nan

            rows.append(
                row
            )

        df_mean = pd.DataFrame(
            rows
        )

    else:

        print(
            "Using simple means..."
        )

        df_mean = (

            df_merge

            .groupby(
                group_cols
            )[mean_vars]

            .mean()

            .reset_index()

        )

    # =================================================
    # FINAL CHECK
    # =================================================

    for spei_name in [
        "SPEI3_Hargreaves",
        "SPEI3_Makkink"
    ]:

        if spei_name in df_mean.columns:

            n = (
                df_mean[spei_name]
                .notna()
                .sum()
            )

            print(
                f"{spei_name}: "
                f"{n} non-missing department-year values"
            )

    # =================================================
    # SAVE
    # =================================================

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

