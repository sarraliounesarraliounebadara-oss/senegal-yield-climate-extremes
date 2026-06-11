# =========================================
# BUILD MIRCA CROP MASKS
# WITHOUT INTERPOLATION
# =========================================

import xarray as xr
import geopandas as gpd
import rioxarray
import numpy as np
import pandas as pd
import os
import unicodedata

# =========================================
# PATHS
# =========================================

SHAPEFILE = (
    "/media/sarr/01DC5DE9D15E8CF0/"
    "departments/agro_data/SEN_adm/SEN_adm2.shp"
)

MIRCA_DIR = (
    "/media/sarr/01DC5DE9D15E8CF0/MIRCA"
)

OUTDIR = (
    "/media/sarr/01DC5DE7BBEA1120/"
    "Donnees/mirca_masks"
)

os.makedirs(OUTDIR, exist_ok=True)

# =========================================
# MIRCA FILES
# =========================================

mirca_files = {

    "maize":
    "crop_02_rainfed.nc",

    "sorghum":
    "crop_05_rainfed.nc",

    "millet":
    "crop_06_rainfed.nc",

    "groundnut":
    "crop_12_rainfed.nc"
}

# =========================================
# STUDY DEPARTMENTS
# =========================================

study_departments = [

    "bakel",
    "bambey",
    "bignona",
    "diourbel",
    "fatick",
    "foundiougne",
    "gossas",
    "kaffrine",
    "kaolack",
    "kebemer",
    "kedougou",
    "kolda",
    "linguere",
    "louga",
    "mbacke",
    "mbour",
    "nioro_du_rip",
    "oussouye",
    "sedhiou",
    "tambacounda",
    "thies",
    "tivaouane",
    "velingara",
    "ziguinchor"
]

# =========================================
# MINIMUM CULTIVATED AREA
# =========================================

MIN_AREA = 0

# =========================================
# CLEAN FUNCTION
# =========================================

def clean_text(x):

    if pd.isna(x):
        return x

    x = str(x).lower().strip()

    x = unicodedata.normalize(
        "NFKD",
        x
    ).encode(
        "ascii",
        "ignore"
    ).decode("utf-8")

    return x.replace(" ", "_")

# =========================================
# READ SHAPEFILE
# =========================================

gdf = gpd.read_file(
    SHAPEFILE
)

gdf = gdf.to_crs(
    "EPSG:4326"
)

gdf["department"] = (

    gdf["NAME_2"]
    .apply(clean_text)
    .str.strip("_")

)

# =========================================
# KEEP ONLY STUDY AREA
# =========================================

gdf = gdf[
    gdf["department"].isin(
        study_departments
    )
]

print(
    "Departments:",
    len(gdf)
)

# =========================================
# LOOP CROPS
# =========================================

for crop, filename in mirca_files.items():

    print("\n==============================")
    print("Crop:", crop)

    path = os.path.join(
        MIRCA_DIR,
        filename
    )

    print("Opening:", path)

    # =====================================
    # OPEN DATASET
    # =====================================

    ds = xr.open_dataset(path)

    da = ds["crop_area"]

    # =====================================
    # MEAN OVER MONTHS
    # =====================================

    print("Computing mean growing area...")

    da = da.mean(
        dim="month",
        skipna=True
    )

    # =====================================
    # RENAME FOR RIOXARRAY
    # =====================================

    da = da.rename({

        "lon": "x",
        "lat": "y"

    })

    # =====================================
    # DEFINE SPATIAL DIMS
    # =====================================

    da = da.rio.set_spatial_dims(

        x_dim="x",
        y_dim="y"

    )

    da = da.rio.write_crs(
        "EPSG:4326"
    )

    # =====================================
    # REMOVE NEGATIVE VALUES
    # =====================================

    da = da.where(
        da > 0
    )

    # =====================================
    # LOOP DEPARTMENTS
    # =====================================

    for _, row in gdf.iterrows():

        dept = row["department"]

        print(
            "Department:",
            dept
        )

        geom = [row.geometry]

        # =================================
        # CLIP
        # =================================

        try:

            clipped = da.rio.clip(

                geom,
                gdf.crs,
                drop=True

            )

        except Exception as e:

            print(
                "Clip error:",
                str(e)
            )

            continue

        # =================================
        # TO DATAFRAME
        # =================================

        df = clipped.to_dataframe(
            name="crop_area"
        ).reset_index()

        # =================================
        # RENAME COORDS
        # =================================

        df = df.rename(columns={

            "x": "lon",
            "y": "lat"

        })

        # =================================
        # REMOVE NAN
        # =================================

        df = df.dropna()

        # =================================
        # KEEP CULTIVATED PIXELS
        # =================================

        df = df[
            df["crop_area"] >= MIN_AREA
        ]

        if df.empty:

            print(
                "No cultivated pixels"
            )

            continue

        # =================================
        # COMPUTE WEIGHTS
        # =================================

        total = df["crop_area"].sum()

        df["weight"] = (

            df["crop_area"]
            / total

        )

        # =================================
        # ADD METADATA
        # =================================

        df["crop"] = crop

        df["department"] = dept

        # =================================
        # KEEP COLUMNS
        # =================================

        df = df[

            [

                "department",
                "crop",
                "lon",
                "lat",
                "crop_area",
                "weight"

            ]

        ]

        # =================================
        # SORT
        # =================================

        df = df.sort_values(
            ["lat", "lon"]
        )

        # =================================
        # SAVE
        # =================================

        outfile = os.path.join(

            OUTDIR,

            f"{crop}_{dept}_mask.csv"

        )

        df.to_csv(

            outfile,

            index=False

        )

        print(
            "✔ Saved:",
            outfile,
            "| pixels:",
            len(df)
        )

    ds.close()

print("\n✅ ALL MASKS COMPLETED")#!/bin/bash
set -e

# --------------------------------------------------
# TMP (évite /home plein)
# --------------------------------------------------
export TMPDIR="/media/sarr/01DC5DE9D15E8CF0/tmp_climate"
export CDO_TMPDIR="/media/sarr/01DC5DE9D15E8CF0/tmp_climate"

NCPU=4

BASE="/media/sarr/01DC5DE9D15E8CF0/departments/climate_data"

CHIRPS="$BASE/CHIRPS_daily_2000-2013_sen.nc"
ERA5_TMAX="$BASE/ERA5_Tmax_2000-2013.nc"
ERA5_TMIN="$BASE/ERA5_Tmin_2000-2013.nc"

CMIP6="$BASE/cmip6"

MASKDIR="/media/sarr/01DC5DE7BBEA1120/Donnees/python-script/crop_masks_nc"

OUT="/media/sarr/01DC5DE9D15E8CF0/climate_data_in_mask"
TMP="/media/sarr/01DC5DE9D15E8CF0/tmp_climate"

mkdir -p "$TMP"
mkdir -p "$OUT"

# ✅ 4 cultures (corrigé)
CROPS=("maize" "millet" "sorghum" "groundnut")

MODELS=(
ACCESS-CM2
ACCESS-ESM1-5
BCC-CSM2-MR
CNRM-CM6-1
FGOALS-g3
GFDL-ESM4
MIROC6
MPI-ESM1-2-LR
MRI-ESM2-0
CCCma
)

SCENARIOS=("ssp245" "ssp585")

# ✅ GRID robuste (basé sur maize)
GRID=$(ls "$MASKDIR"/mask_maize_*_4km.nc | head -n 1)

echo "GRID utilisé: $GRID"

echo "--------------------------------"
echo "REMAPPING OBS DATA"
echo "--------------------------------"

[ ! -f "$TMP/chirps_remap.nc" ] && \
cdo -O remapbil,"$GRID" -selmon,4,5,6,7,8,9,10 "$CHIRPS" "$TMP/chirps_remap.nc"

[ ! -f "$TMP/tmax_remap.nc" ] && \
cdo -O remapbil,"$GRID" -selmon,4,5,6,7,8,9,10 "$ERA5_TMAX" "$TMP/tmax_remap.nc"

[ ! -f "$TMP/tmin_remap.nc" ] && \
cdo -O remapbil,"$GRID" -selmon,4,5,6,7,8,9,10 "$ERA5_TMIN" "$TMP/tmin_remap.nc"

echo "--------------------------------"
echo "START PIPELINE"
echo "--------------------------------"

for crop in "${CROPS[@]}"
do

OBS_OUT="$OUT/$crop/observation"
CMIP_OUT="$OUT/$crop/cmip6"

mkdir -p "$OBS_OUT/chirps"
mkdir -p "$OBS_OUT/ERA5"
mkdir -p "$CMIP_OUT/historical"
mkdir -p "$CMIP_OUT/near_future"
mkdir -p "$CMIP_OUT/far_future"

####################################
# OBSERVATIONS
####################################

echo "OBSERVATIONS $crop"

for mask in "$MASKDIR"/mask_${crop}_*_4km.nc
do

name=$(basename "$mask" .nc)
MASK_TMP="$TMP/mask_${name}.nc"

cdo -O remapnn,"$TMP/chirps_remap.nc" "$mask" "$MASK_TMP"

OUTFILE="$OBS_OUT/chirps/chirps_${crop}_historical_${name}_fldmean.nc"
[ ! -f "$OUTFILE" ] && \
cdo -O fldmean -ifthen "$MASK_TMP" "$TMP/chirps_remap.nc" "$OUTFILE"

OUTFILE="$OBS_OUT/ERA5/era5_tmax_${crop}_historical_${name}_fldmean.nc"
[ ! -f "$OUTFILE" ] && \
cdo -O fldmean -ifthen "$MASK_TMP" "$TMP/tmax_remap.nc" "$OUTFILE"

OUTFILE="$OBS_OUT/ERA5/era5_tmin_${crop}_historical_${name}_fldmean.nc"
[ ! -f "$OUTFILE" ] && \
cdo -O fldmean -ifthen "$MASK_TMP" "$TMP/tmin_remap.nc" "$OUTFILE"

rm -f "$MASK_TMP"

done

####################################
# CMIP6
####################################

for model in "${MODELS[@]}"
do
for scen in "${SCENARIOS[@]}"
do

echo "MODEL $model $scen"

PRFILE=$(ls "$CMIP6/pr/${model}_${scen}"*prAdjust_CDFt-L-1V-0L.nc 2>/dev/null | head -n 1)
TMAXFILE=$(ls "$CMIP6/tasmax/${model}"*"${scen}"*tasmaxAdjust_CDFt-L-1V-0L.nc 2>/dev/null | head -n 1)
TMINFILE=$(ls "$CMIP6/tasmin/${model}"*"${scen}"*tasminAdjust_CDFt-L-1V-0L.nc 2>/dev/null | head -n 1)

# ✅ skip si fichier absent
[ -z "$PRFILE" ] && continue
[ -z "$TMAXFILE" ] && continue
[ -z "$TMINFILE" ] && continue

TMP_PR="$TMP/pr_${model}_${scen}.nc"
TMP_TMAX="$TMP/tmax_${model}_${scen}.nc"
TMP_TMIN="$TMP/tmin_${model}_${scen}.nc"

# PR
if [ ! -f "$TMP_PR" ]; then
cdo -O selname,prAdjust "$PRFILE" "$TMP/pr1.nc"
cdo -O setgrid,"$GRID" "$TMP/pr1.nc" "$TMP/pr2.nc"
cdo -O selmon,4,5,6,7,8,9,10 "$TMP/pr2.nc" "$TMP/pr1.nc"
cdo -O remapbil,"$GRID" "$TMP/pr1.nc" "$TMP_PR"
rm -f "$TMP/pr1.nc" "$TMP/pr2.nc"
fi

# TMAX
if [ ! -f "$TMP_TMAX" ]; then
cdo -O selname,tasmaxAdjust "$TMAXFILE" "$TMP/tmax1.nc"
cdo -O setgrid,"$GRID" "$TMP/tmax1.nc" "$TMP/tmax2.nc"
cdo -O selmon,4,5,6,7,8,9,10 "$TMP/tmax2.nc" "$TMP/tmax1.nc"
cdo -O remapbil,"$GRID" "$TMP/tmax1.nc" "$TMP_TMAX"
rm -f "$TMP/tmax1.nc" "$TMP/tmax2.nc"
fi

# TMIN
if [ ! -f "$TMP_TMIN" ]; then
cdo -O selname,tasminAdjust "$TMINFILE" "$TMP/tmin1.nc"
cdo -O setgrid,"$GRID" "$TMP/tmin1.nc" "$TMP/tmin2.nc"
cdo -O selmon,4,5,6,7,8,9,10 "$TMP/tmin2.nc" "$TMP/tmin1.nc"
cdo -O remapbil,"$GRID" "$TMP/tmin1.nc" "$TMP_TMIN"
rm -f "$TMP/tmin1.nc" "$TMP/tmin2.nc"
fi

for mask in "$MASKDIR"/mask_${crop}_*_4km.nc
do

name=$(basename "$mask" .nc)
MASK_TMP="$TMP/mask_${name}.nc"

cdo -O remapnn,"$TMP_PR" "$mask" "$MASK_TMP"

# HISTORICAL
cdo -O fldmean -ifthen "$MASK_TMP" -selyear,2000/2013 "$TMP_PR" \
"$CMIP_OUT/historical/pr_${model}_${crop}_${name}.nc"

# NEAR
cdo -O fldmean -ifthen "$MASK_TMP" -selyear,2036/2065 "$TMP_PR" \
"$CMIP_OUT/near_future/pr_${model}_${scen}_${crop}_${name}.nc"

# FAR
cdo -O fldmean -ifthen "$MASK_TMP" -selyear,2071/2100 "$TMP_PR" \
"$CMIP_OUT/far_future/pr_${model}_${scen}_${crop}_${name}.nc"

rm -f "$MASK_TMP"

done

rm -f "$TMP_PR" "$TMP_TMAX" "$TMP_TMIN"

done
done
done

echo "--------------------------------"
echo "CLEAN TMP"
echo "--------------------------------"

rm -f "$TMP"/*.txt

echo "--------------------------------"
echo "EXTRACTION TERMINÉE"
echo "--------------------------------"
