#!/bin/bash
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
