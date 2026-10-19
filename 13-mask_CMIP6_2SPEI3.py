#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

EXTREMES_DIR = (
    f"{BASE}/departments/climate_data/extremes"
)

# Grilles MIRCA Sénégal
GRID_DIR = (
    f"{BASE}/results_agro/results_agro2"
)

# Sorties utilisées ensuite par le Random Forest
OUT_DIR = (
    f"{BASE}/departments/climate_data/extremes/cmip6_crop_tables"
)

os.makedirs(OUT_DIR, exist_ok=True)

# Nouveau SPEI3-Makkink CMIP6
MAKKINK_ROOT = (
    f"{BASE}/departments/climate_data/"
    "cmip6/spei3_makkink_fast"
)

MAKKINK_HIST_DIR = os.path.join(
    MAKKINK_ROOT,
    "historical"
)

MAKKINK_SSP245_DIR = os.path.join(
    MAKKINK_ROOT,
    "ssp245"
)

MAKKINK_SSP585_DIR = os.path.join(
    MAKKINK_ROOT,
    "ssp585"
)

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
# MODELS TO PROCESS
# =====================================================
# IMPORTANT: this list matches the SPEI3-Makkink files
# currently present in spei3_makkink_fast. Extra legacy
# extremes files in EXTREMES_DIR are deliberately ignored.
MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CCCma",
    "CNRM-ESM2-1",
    "FGOALS-g3",
    "GFDL-ESM4",
    "MIROC6",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
]

# =====================================================
# VARIABLES
# =====================================================

# SPEI3 de l'ancien fichier extremes = Hargreaves.
# Il sera renommé SPEI3_Hargreaves.
# SPEI3_Makkink sera ajouté depuis les fichiers séparés.

vars_keep = [

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

        # Garder lon/lat en float64 pour sécuriser les merges
        if col not in ["lon", "lat"]:

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
# DETECT COORDS
# =====================================================

def detect_lon_lat(ds):

    lon_name = None
    lat_name = None

    for c in ds.coords:

        cname = c.lower()

        if "lon" in cname:
            lon_name = c

        if "lat" in cname:
            lat_name = c

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
            "Could not detect lon/lat. "
            f"coords={list(ds.coords)}, dims={list(ds.dims)}"
        )

    return lon_name, lat_name

# =====================================================
# MAKKINK FILE PATH
# =====================================================

def get_makkink_file(
    model,
    scenario=None,
    period=None
):

    if scenario is None:

        return os.path.join(
            MAKKINK_HIST_DIR,
            f"{model}_historical_spei3_makkink_1984-2013.nc"
        )

    if scenario == "ssp245":

        base_dir = MAKKINK_SSP245_DIR

    elif scenario == "ssp585":

        base_dir = MAKKINK_SSP585_DIR

    else:

        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    if period == "near":

        years = "2036-2065"

    elif period == "far":

        years = "2071-2100"

    else:

        raise ValueError(
            f"Unknown future period: {period}"
        )

    return os.path.join(
        base_dir,
        f"{model}_{scenario}_{period}_"
        f"spei3_makkink_{years}.nc"
    )

# =====================================================
# LOAD MAKKINK AS DATAFRAME
# =====================================================

def load_makkink_dataframe(
    model,
    scenario=None,
    period=None
):

    makkink_file = get_makkink_file(
        model=model,
        scenario=scenario,
        period=period
    )

    print("\nMakkink file:")
    print(makkink_file)

    if not os.path.exists(
        makkink_file
    ):

        raise FileNotFoundError(
            f"Fichier SPEI3-Makkink manquant : {makkink_file}"
        )

    ds_m = xr.open_dataset(
        makkink_file
    )

    try:

        lon_m, lat_m = detect_lon_lat(
            ds_m
        )

        # Sécurité : un SPEI3 annuel correct ne doit pas
        # conserver une dimension indépendante "month".
        for candidate in ["SPEI3", "SPEI3_Makkink"]:
            if candidate in ds_m.data_vars and "month" in ds_m[candidate].dims:
                raise RuntimeError(
                    "Le fichier Makkink contient une dimension 'month' indésirable : "
                    f"{makkink_file}; dims={ds_m[candidate].dims}"
                )

        if "SPEI3" in ds_m.data_vars:

            var_m = "SPEI3"

        elif "SPEI3_Makkink" in ds_m.data_vars:

            var_m = "SPEI3_Makkink"

        else:

            raise KeyError(
                "No SPEI3 variable in "
                f"{makkink_file}. "
                f"Variables={list(ds_m.data_vars)}"
            )

        df_m = (

            ds_m[
                [var_m]
            ]

            .to_dataframe()

            .reset_index()

        )

    finally:

        ds_m.close()

    df_m.rename(

        columns={
            lon_m: "lon",
            lat_m: "lat",
            var_m: "SPEI3_Makkink"
        },

        inplace=True

    )

    if "time" not in df_m.columns:

        raise KeyError(
            f"No time coordinate in {makkink_file}"
        )

    df_m["year"] = extract_year(
        df_m["time"]
    )

    # Harmonisation numérique des coordonnées
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

    df_m = df_m[
        [
            "lon",
            "lat",
            "year",
            "SPEI3_Makkink"
        ]
    ]

    # Sécurité : une seule valeur par pixel et année
    duplicates = df_m.duplicated(
        subset=[
            "lon",
            "lat",
            "year"
        ]
    ).sum()

    if duplicates > 0:

        print(
            "WARNING:",
            duplicates,
            "duplicate Makkink pixel-year rows."
        )

        df_m = (
            df_m
            .groupby(
                [
                    "lon",
                    "lat",
                    "year"
                ],
                as_index=False
            )["SPEI3_Makkink"]
            .mean()
        )

    print(
        "Makkink years:",
        df_m["year"].min(),
        "-",
        df_m["year"].max()
    )

    return df_m

# =====================================================
# ADD MAKKINK TO EXTREMES DATAFRAME
# =====================================================

def add_makkink_to_extremes(
    df_ext,
    model,
    scenario=None,
    period=None
):

    df_m = load_makkink_dataframe(
        model=model,
        scenario=scenario,
        period=period
    )

    if df_m is None:

        df_ext["SPEI3_Makkink"] = np.nan

        return df_ext

    # -------------------------------------------------
    # First attempt: exact rounded lon/lat/year merge
    # -------------------------------------------------

    n_before = len(
        df_ext
    )

    df_merged = pd.merge(

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

    if len(df_merged) != n_before:

        raise RuntimeError(
            "Number of rows changed after "
            "Makkink merge."
        )

    n_valid = (
        df_merged[
            "SPEI3_Makkink"
        ]
        .notna()
        .sum()
    )

    print(
        "Exact Makkink matches:",
        n_valid,
        "/",
        len(df_merged)
    )

    # -------------------------------------------------
    # If grids are not exactly identical, use nearest
    # Makkink pixel for each climate pixel, then merge
    # by mapped coordinates + year.
    # -------------------------------------------------

    if n_valid == 0:

        print(
            "No exact coordinate matches. "
            "Using nearest Makkink grid..."
        )

        # Remove unsuccessful column
        df_ext2 = df_ext.copy()

        m_pixels = (
            df_m[
                ["lon", "lat"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        e_pixels = (
            df_ext2[
                ["lon", "lat"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        m_tree = cKDTree(
            m_pixels[
                ["lon", "lat"]
            ].values
        )

        dist_m, idx_m = m_tree.query(

            e_pixels[
                ["lon", "lat"]
            ].values,

            k=1

        )

        print(
            "Makkink grid max distance:",
            dist_m.max()
        )

        print(
            "Makkink grid mean distance:",
            dist_m.mean()
        )

        nearest_m = m_pixels.iloc[
            idx_m
        ].reset_index(
            drop=True
        )

        mapping = e_pixels.copy()

        mapping["lon_m"] = (
            nearest_m["lon"].values
        )

        mapping["lat_m"] = (
            nearest_m["lat"].values
        )

        df_ext2 = pd.merge(

            df_ext2,
            mapping,

            on=[
                "lon",
                "lat"
            ],

            how="left",
            validate="many_to_one"

        )

        df_m2 = df_m.rename(
            columns={
                "lon": "lon_m",
                "lat": "lat_m"
            }
        )

        df_merged = pd.merge(

            df_ext2,
            df_m2,

            on=[
                "lon_m",
                "lat_m",
                "year"
            ],

            how="left",
            validate="many_to_one"

        )

        df_merged.drop(
            columns=[
                "lon_m",
                "lat_m"
            ],
            inplace=True
        )

        n_valid = (
            df_merged[
                "SPEI3_Makkink"
            ]
            .notna()
            .sum()
        )

        print(
            "Nearest-grid Makkink matches:",
            n_valid,
            "/",
            len(df_merged)
        )

    del df_m
    gc.collect()

    return df_merged

# =====================================================
# PROCESS FUNCTION
# =====================================================

def process_extremes_file(
    ncfile,
    df_grid,
    model,
    scenario=None,
    future=False,
    period=None
):

    print(
        f"\nProcessing: "
        f"{os.path.basename(ncfile)}"
    )

    # =================================================
    # OPEN DATASET
    # =================================================

    ds = xr.open_dataset(
        ncfile
    )

    # =================================================
    # DETECT COORDS
    # =================================================

    try:

        lon_name, lat_name = detect_lon_lat(
            ds
        )

    except Exception as exc:

        print(
            "Could not detect lon/lat:",
            repr(exc)
        )

        ds.close()

        return None

    # =================================================
    # RENAME ORIGINAL SPEI3 -> HARGREAVES
    # =================================================

    if "SPEI3" in ds.data_vars:

        ds = ds.rename(
            {
                "SPEI3":
                "SPEI3_Hargreaves"
            }
        )

    elif (
        "SPEI3_Hargreaves"
        not in ds.data_vars
    ):

        print(
            "WARNING: no SPEI3 "
            "Hargreaves found."
        )

    # =================================================
    # KEEP VARIABLES
    # =================================================

    # Makkink is loaded separately below.

    existing_vars = [

        v for v in vars_keep

        if (
            v != "SPEI3_Makkink"
            and v in ds.data_vars
        )

    ]

    print(
        "\nVariables found in extremes:"
    )

    print(
        existing_vars
    )

    if len(
        existing_vars
    ) == 0:

        print(
            "No climate variables."
        )

        ds.close()

        return None

    # =================================================
    # DATAFRAME
    # =================================================

    df_ext = (

        ds[
            existing_vars
        ]

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
    # COORDINATE NORMALIZATION
    # =================================================

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

    # =================================================
    # MEMORY OPT
    # =================================================

    df_ext = optimize_dataframe(
        df_ext
    )

    # =================================================
    # ADD SPEI3-MAKKINK
    # =================================================

    df_ext = add_makkink_to_extremes(

        df_ext=df_ext,
        model=model,
        scenario=scenario,
        period=period

    )

    print(
        "\nVariables after adding Makkink:"
    )

    print(
        [
            v for v in vars_keep
            if v in df_ext.columns
        ]
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
        pixels[
            ["lon", "lat"]
        ].values
    )

    # =================================================
    # GRID COPY + COORDINATE NORMALIZATION
    # =================================================

    df_tmp = df_grid.copy()

    df_tmp["lon"] = (
        df_tmp["lon"]
        .astype("float64")
        .round(6)
    )

    df_tmp["lat"] = (
        df_tmp["lat"]
        .astype("float64")
        .round(6)
    )

    # =================================================
    # NEAREST PIXELS
    # =================================================

    dist, idx = tree.query(

        df_tmp[
            ["lon", "lat"]
        ].values,

        k=1

    )

    print(
        "Max distance:",
        dist.max()
    )

    print(
        "Mean distance:",
        dist.mean()
    )

    if dist.max() > 0.05:

        print(
            "WARNING: MIRCA-climate "
            "distance > 0.05°"
        )

    nearest = pixels.iloc[
        idx
    ].reset_index(
        drop=True
    )

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

        print(
            "\nHistorical merge"
        )

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
    # DO NOT MERGE ON MIRCA YEAR
    else:

        print(
            "\nFuture merge"
        )

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
    # =================================================

    mean_vars = []

    for col in vars_keep:

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

        row[
            "model"
        ] = model

        if scenario is not None:

            row[
                "scenario"
            ] = scenario

        rows.append(
            row
        )

    df_mean = pd.DataFrame(
        rows
    )

    # =================================================
    # NaN CHECK
    # =================================================

    print(
        "\nNaN ratio:"
    )

    print(
        df_mean[
            mean_vars
        ]
        .isna()
        .mean()
    )

    # SPEI diagnostic
    for spei_name in [
        "SPEI3_Hargreaves",
        "SPEI3_Makkink"
    ]:

        if spei_name in df_mean.columns:

            n_ok = (
                df_mean[
                    spei_name
                ]
                .notna()
                .sum()
            )

            print(
                f"{spei_name}: "
                f"{n_ok} non-missing "
                f"department-year values"
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
# OUTPUT COLUMN ORDER
# =====================================================

def order_output_columns(df):

    first = [
        c for c in [
            "department",
            "year",
            "scenario",
            "model"
        ]
        if c in df.columns
    ]

    climate = [
        c for c in vars_keep
        if c in df.columns
    ]

    rest = [
        c for c in df.columns
        if c not in first + climate
    ]

    return df[first + climate + rest]


# =====================================================
# OUTPUT VALIDATION
# =====================================================

def validate_two_spei_columns(df, label):

    required = [
        "SPEI3_Hargreaves",
        "SPEI3_Makkink"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label}: colonnes SPEI3 manquantes : {missing}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    for col in required:

        n_ok = int(df[col].notna().sum())
        n_total = len(df)

        print(
            f"{label} - {col}: "
            f"{n_ok}/{n_total} valeurs valides"
        )

        if n_ok == 0:
            raise RuntimeError(
                f"{label}: {col} est entièrement NaN."
            )

    # Diagnostic by model: never silently drop a model.
    if "model" in df.columns:
        print("\nDiagnostic SPEI3_Makkink par modèle:")
        bad_models = []

        for model, g in df.groupby("model", sort=True):
            n_ok = int(g["SPEI3_Makkink"].notna().sum())
            n_tot = len(g)
            print(f"  {model:20s}: {n_ok}/{n_tot} valides")

            if n_ok == 0:
                bad_models.append(model)

        if bad_models:
            print(
                "WARNING: SPEI3_Makkink entièrement NaN pour: "
                + ", ".join(bad_models)
            )
            print(
                "Ces lignes sont conservées dans le CSV; "
                "aucune valeur n'est inventée ou imputée par le masque."
            )

    print(
        f"{label}: OK -> "
        "SPEI3_Hargreaves + SPEI3_Makkink présents"
    )


# =====================================================
# PREFLIGHT: EXACT INPUT FILES USED BY THIS MASK
# =====================================================

def expected_extremes_file(model, scenario=None, period=None):

    if scenario is None:
        return os.path.join(
            EXTREMES_DIR,
            "historical",
            f"{model}_historical_extremes.nc"
        )

    return os.path.join(
        EXTREMES_DIR,
        scenario,
        f"{model}_{scenario}_{period}_extremes.nc"
    )


def preflight_inputs():

    missing = []

    for model in MODELS:

        # Historical extremes + historical Makkink
        f_ext = expected_extremes_file(model)
        f_mak = get_makkink_file(model)

        if not os.path.exists(f_ext):
            missing.append(f_ext)

        if not os.path.exists(f_mak):
            missing.append(f_mak)

        # Near/far, SSP245/585
        for scenario in scenarios:
            for period in ["near", "far"]:

                f_ext = expected_extremes_file(
                    model,
                    scenario=scenario,
                    period=period
                )

                f_mak = get_makkink_file(
                    model,
                    scenario=scenario,
                    period=period
                )

                if not os.path.exists(f_ext):
                    missing.append(f_ext)

                if not os.path.exists(f_mak):
                    missing.append(f_mak)

    if missing:
        print("\nERREUR - fichiers d'entrée manquants:")
        for f in missing:
            print("  ", f)

        raise FileNotFoundError(
            f"{len(missing)} fichier(s) requis manquant(s). "
            "Le masque n'a rien écrit."
        )

    print("\nPrécontrôle fichiers: OK")
    print("Modèles traités:", len(MODELS))
    for model in MODELS:
        print("  -", model)


preflight_inputs()

# =====================================================
# LOOP CROPS
# =====================================================

print("\n====================================")
print("CONFIGURATION")
print("====================================")
print("EXTREMES_DIR :", EXTREMES_DIR)
print("GRID_DIR     :", GRID_DIR)
print("MAKKINK_ROOT :", MAKKINK_ROOT)
print("OUT_DIR      :", OUT_DIR)
print("N_MODELS     :", len(MODELS))

for crop in crops:

    print(
        "\n===================================="
    )

    print(
        crop
    )

    print(
        "===================================="
    )

    # =================================================
    # LOAD GRID
    # =================================================

    grid_file = (
        f"{GRID_DIR}/{crop}_grids.csv"
    )

    if not os.path.exists(
        grid_file
    ):

        print(
            "Missing:"
        )

        print(
            grid_file
        )

        continue

    df_grid = pd.read_csv(
        grid_file
    )

    if df_grid.empty:
        raise RuntimeError(
            f"Grille MIRCA vide : {grid_file}"
        )

    if "department" not in df_grid.columns:
        raise RuntimeError(
            f"La grille MIRCA n'est pas une grille Sénégal valide : {grid_file}. "
            f"Colonnes={df_grid.columns.tolist()}"
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

    df_grid = optimize_dataframe(
        df_grid
    )

    # =================================================
    # HISTORICAL
    # =================================================

    hist_all = []

    hist_dir = (
        f"{EXTREMES_DIR}/historical"
    )

    # Only the 10 models listed in MODELS are processed.
    # Extra/legacy NetCDF files in the directory are ignored.
    for model in MODELS:

        ncfile = (
            f"{hist_dir}/{model}_historical_extremes.nc"
        )

        df_model = process_extremes_file(

            ncfile=ncfile,
            df_grid=df_grid,
            model=model,
            scenario=None,
            future=False,
            period=None

        )

        if df_model is not None:

            hist_all.append(
                df_model
            )

    # =================================================
    # SAVE HIST
    # =================================================

    if len(
        hist_all
    ) > 0:

        hist_df = pd.concat(
            hist_all,
            ignore_index=True
        )

        # SORTIE INCHANGÉE
        hist_out = (
            f"{OUT_DIR}/"
            f"{crop}_historical_extremes.csv"
        )

        hist_df = order_output_columns(hist_df)

        validate_two_spei_columns(
            hist_df,
            f"{crop} historical"
        )

        hist_df.to_csv(
            hist_out,
            index=False
        )

        print(
            "\nSaved:"
        )

        print(
            hist_out
        )

        del hist_df
        gc.collect()

    # =================================================
    # NEAR FUTURE
    # =================================================

    near_all = []

    for scenario in scenarios:

        scen_dir = (
            f"{EXTREMES_DIR}/{scenario}"
        )

        for model in MODELS:

            ncfile = (
                f"{scen_dir}/{model}_{scenario}_near_extremes.nc"
            )

            df_model = process_extremes_file(

                ncfile=ncfile,
                df_grid=df_grid,
                model=model,
                scenario=scenario,
                future=True,
                period="near"

            )

            if df_model is not None:

                near_all.append(
                    df_model
                )

    # =================================================
    # SAVE NEAR
    # =================================================

    if len(
        near_all
    ) > 0:

        near_df = pd.concat(
            near_all,
            ignore_index=True
        )

        # SORTIE INCHANGÉE
        near_out = (
            f"{OUT_DIR}/"
            f"{crop}_near_future_extremes.csv"
        )

        near_df = order_output_columns(near_df)

        validate_two_spei_columns(
            near_df,
            f"{crop} near_future"
        )

        near_df.to_csv(
            near_out,
            index=False
        )

        print(
            "\nSaved:"
        )

        print(
            near_out
        )

        del near_df
        gc.collect()

    # =================================================
    # FAR FUTURE
    # =================================================

    far_all = []

    for scenario in scenarios:

        scen_dir = (
            f"{EXTREMES_DIR}/{scenario}"
        )

        for model in MODELS:

            ncfile = (
                f"{scen_dir}/{model}_{scenario}_far_extremes.nc"
            )

            df_model = process_extremes_file(

                ncfile=ncfile,
                df_grid=df_grid,
                model=model,
                scenario=scenario,
                future=True,
                period="far"

            )

            if df_model is not None:

                far_all.append(
                    df_model
                )

    # =================================================
    # SAVE FAR
    # =================================================

    if len(
        far_all
    ) > 0:

        far_df = pd.concat(
            far_all,
            ignore_index=True
        )

        # SORTIE INCHANGÉE
        far_out = (
            f"{OUT_DIR}/"
            f"{crop}_far_future_extremes.csv"
        )

        far_df = order_output_columns(far_df)

        validate_two_spei_columns(
            far_df,
            f"{crop} far_future"
        )

        far_df.to_csv(
            far_out,
            index=False
        )

        print(
            "\nSaved:"
        )

        print(
            far_out
        )

        del far_df
        gc.collect()

# =====================================================
# END
# =====================================================

print(
    "\n===================================="
)

print(
    "DONE"
)

print(
    "===================================="
)

