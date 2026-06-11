# =========================================
# CONVERT MIRCA PIXEL CSV MASKS TO NETCDF
# =========================================

import xarray as xr
import pandas as pd
import numpy as np
import glob
import os

# =========================================
# PATHS
# =========================================

CSV_DIR = (
    "/media/sarr/01DC5DE7BBEA1120/"
    "Donnees/mirca_masks"
)

OUTDIR = (
    "/media/sarr/01DC5DE7BBEA1120/"
    "Donnees/mirca_masks_nc"
)

os.makedirs(OUTDIR, exist_ok=True)

# =========================================
# GET ALL CSV FILES
# =========================================

files = glob.glob(
    os.path.join(CSV_DIR, "*.csv")
)

print(
    "Files found:",
    len(files)
)

# =========================================
# LOOP FILES
# =========================================

for f in files:

    print("\n==========================")
    print("Processing:")
    print(os.path.basename(f))

    # =====================================
    # READ CSV
    # =====================================

    df = pd.read_csv(f)

    if df.empty:

        print("Empty file")

        continue

    # =====================================
    # UNIQUE COORDS
    # =====================================

    lon = np.sort(
        df["lon"].unique()
    )

    lat = np.sort(
        df["lat"].unique()
    )

    # =====================================
    # EMPTY GRID
    # =====================================

    mask = np.full(

        (
            len(lat),
            len(lon)
        ),

        np.nan,

        dtype=np.float32

    )

    weight = np.full(

        (
            len(lat),
            len(lon)
        ),

        np.nan,

        dtype=np.float32

    )

    # =====================================
    # BUILD LOOKUP
    # =====================================

    lon_index = {
        v: i for i, v in enumerate(lon)
    }

    lat_index = {
        v: i for i, v in enumerate(lat)
    }

    # =====================================
    # FILL GRID
    # =====================================

    for _, row in df.iterrows():

        i = lat_index[row["lat"]]

        j = lon_index[row["lon"]]

        mask[i, j] = row["crop_area"]

        weight[i, j] = row["weight"]

    # =====================================
    # DATAARRAYS
    # =====================================

    da_mask = xr.DataArray(

        mask,

        coords={

            "lat": lat,
            "lon": lon

        },

        dims=(

            "lat",
            "lon"

        ),

        name="crop_area"

    )

    da_weight = xr.DataArray(

        weight,

        coords={

            "lat": lat,
            "lon": lon

        },

        dims=(

            "lat",
            "lon"

        ),

        name="weight"

    )

    # =====================================
    # ATTRIBUTES
    # =====================================

    da_mask.attrs["units"] = "ha"

    da_mask.attrs["description"] = (
        "MIRCA cultivated area"
    )

    da_weight.attrs["description"] = (
        "Normalized crop weights"
    )

    # =====================================
    # DATASET
    # =====================================

    ds = xr.Dataset({

        "crop_area": da_mask,
        "weight": da_weight

    })

    # =====================================
    # GLOBAL ATTRIBUTES
    # =====================================

    ds.attrs["source"] = "MIRCA2000"

    ds.attrs["type"] = (
        "Department crop mask"
    )

    # =====================================
    # OUTPUT NAME
    # =====================================

    basename = os.path.basename(f)

    outname = basename.replace(
        ".csv",
        ".nc"
    )

    outpath = os.path.join(
        OUTDIR,
        outname
    )

    # =====================================
    # SAVE
    # =====================================

    ds.to_netcdf(outpath)

    print(
        "✔ Saved:",
        outpath
    )

print("\n✅ ALL NETCDF MASKS CREATED")#!/bin/bash
set -e
set -o pipefail
shopt -s nullglob

echo "--------------------------------"
echo "CALCUL EXTREMES CLIMATIQUES"
echo "--------------------------------"

BASE="/media/sarr/01DC5DE9D15E8CF0/climate_data_in_mask"
OUT="/media/sarr/01DC5DE9D15E8CF0/climate_extremes"

# 
CROPS=(maize millet sorghum groundnut)
PERIODS=(historical near_future far_future)

################################
# FONCTION R95pTOT (CORRIGÉE)
################################
calc_r95ptot() {
    PR=$1
    OUTFILE=$2

    TMPDIR=$(mktemp -d)

    # seuil percentile 95 JJAS (correct)
    cdo timpctl,95 -selmon,6,7,8,9 "$PR" \
        -timmin -selmon,6,7,8,9 "$PR" \
        -timmax -selmon,6,7,8,9 "$PR" \
        "$TMPDIR/p95.nc"

    # pluie > p95
    cdo -L ifthen -gt "$PR" "$TMPDIR/p95.nc" "$PR" \
        "$TMPDIR/r95.nc"

    # somme annuelle
    cdo yearsum -selmon,6,7,8,9 "$TMPDIR/r95.nc" "$OUTFILE"

    rm -rf "$TMPDIR"
}

################################
# LOOP PRINCIPALE
################################

for crop in "${CROPS[@]}"
do

echo "=============================="
echo "CROP $crop"

BASECROP="$BASE/$crop"
OUTCROP="$OUT/$crop"

mkdir -p "$OUTCROP/observation"
mkdir -p "$OUTCROP/cmip6/historical"
mkdir -p "$OUTCROP/cmip6/near_future"
mkdir -p "$OUTCROP/cmip6/far_future"

################################
# OBSERVATIONS CHIRPS
################################

for PR in "$BASECROP"/observation/chirps/*.nc
do
[ -f "$PR" ] || continue

name=$(basename "$PR" .nc)

echo "OBS PR $name"

# RX1day
cdo yearmax -selmon,6,7,8,9 "$PR" \
"$OUTCROP/observation/${name}_RX1day.nc"

# CDD
cdo yearmax -consecsum -ltc,1 -selmon,6,7,8,9 "$PR" \
"$OUTCROP/observation/${name}_CDD.nc"

# R20mm
cdo yearsum -gec,20 -selmon,6,7,8,9 "$PR" \
"$OUTCROP/observation/${name}_R20mm.nc"

# PRtotal
cdo yearsum -selmon,6,7,8,9 "$PR" \
"$OUTCROP/observation/${name}_PRtotal.nc"

# R95pTOT
calc_r95ptot "$PR" \
"$OUTCROP/observation/${name}_R95pTOT.nc"

done

################################
# ERA5 TEMPERATURE
################################

for TMAX in "$BASECROP"/observation/ERA5/*tmax*.nc
do
[ -f "$TMAX" ] || continue

name=$(basename "$TMAX" .nc)

echo "OBS TMAX $name"

# TXx
cdo yearmax -selmon,6,7,8,9 "$TMAX" \
"$OUTCROP/observation/${name}_TXx.nc"

# TX35
cdo yearsum -gec,35 -selmon,6,7,8,9 "$TMAX" \
"$OUTCROP/observation/${name}_TX35.nc"

done

################################
# CMIP6
################################

for period in "${PERIODS[@]}"
do

echo "---- PERIOD $period ----"

################################
# PRECIPITATION
################################

for PR in "$BASECROP"/cmip6/"$period"/pr_*.nc
do
[ -f "$PR" ] || continue

name=$(basename "$PR" .nc)

echo "CMIP6 PR $name"

cdo yearmax -selmon,6,7,8,9 "$PR" \
"$OUTCROP/cmip6/$period/${name}_RX1day.nc"

cdo yearmax -consecsum -ltc,1 -selmon,6,7,8,9 "$PR" \
"$OUTCROP/cmip6/$period/${name}_CDD.nc"

cdo yearsum -gec,20 -selmon,6,7,8,9 "$PR" \
"$OUTCROP/cmip6/$period/${name}_R20mm.nc"

cdo yearsum -selmon,6,7,8,9 "$PR" \
"$OUTCROP/cmip6/$period/${name}_PRtotal.nc"

calc_r95ptot "$PR" \
"$OUTCROP/cmip6/$period/${name}_R95pTOT.nc"

done

################################
# TEMPERATURE
################################

for TMAX in "$BASECROP"/cmip6/"$period"/tmax_*.nc
do
[ -f "$TMAX" ] || continue

name=$(basename "$TMAX" .nc)

echo "CMIP6 TMAX $name"

cdo yearmax -selmon,6,7,8,9 "$TMAX" \
"$OUTCROP/cmip6/$period/${name}_TXx.nc"

cdo yearsum -gec,35 -selmon,6,7,8,9 "$TMAX" \
"$OUTCROP/cmip6/$period/${name}_TX35.nc"

done

done

done

echo "--------------------------------"
echo "CALCUL EXTREMES TERMINE"
echo "--------------------------------"
