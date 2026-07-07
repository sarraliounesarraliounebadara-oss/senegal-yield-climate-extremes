#!/bin/bash

set -e
set -o pipefail
shopt -s nullglob

echo "--------------------------------"
echo "CMIP6 EXTREMES JJAS"
echo "--------------------------------"

# ======================================================
# ENVIRONMENT
# ======================================================

export CDO_FILE_SUFFIX=NULL
export CDI_NETCDF_TYPE=nc4
export OMP_NUM_THREADS=1

# ======================================================
# PATHS
# ======================================================

BASE="/media/sarr/01DC5DE9D15E8CF0/departments/climate_data"

OBS_DIR="$BASE/anacim_data"

CMIP6_DIR="$BASE/cmip6"

SPEI_DIR="$BASE/spei3"

OUT="$BASE/extremes"

TMP="/tmp/extremes_jjas"

mkdir -p "$TMP"

mkdir -p "$OUT/observations"
mkdir -p "$OUT/historical"
mkdir -p "$OUT/ssp245"
mkdir -p "$OUT/ssp585"

# ======================================================
# PERIODS
# ======================================================

REF_START="1984-01-01"
REF_END="2013-12-31"

NEAR_START="2036-01-01"
NEAR_END="2065-12-31"

FAR_START="2071-01-01"
FAR_END="2100-12-31"

# ======================================================
# COMPUTE DAILY CLIMATOLOGICAL P95
# ======================================================

compute_p95_reference () {

    infile=$1
    outfile=$2

    tmp_jjas="$TMP/p95_jjas.nc"
    tmp_wet="$TMP/p95_wet.nc"
    tmp_min="$TMP/p95_min.nc"
    tmp_max="$TMP/p95_max.nc"

    echo ">>> Computing climatological P95"

    # JJAS only
    cdo -O -L selmon,6/9 \
        "$infile" \
        "$tmp_jjas"

    # Wet days only (RR >= 1 mm)
    cdo -O -L setrtomiss,-inf,0.999 \
        "$tmp_jjas" \
        "$tmp_wet"

    # Required for ydaypctl
    cdo -O -L ydaymin \
        "$tmp_wet" \
        "$tmp_min"

    cdo -O -L ydaymax \
        "$tmp_wet" \
        "$tmp_max"

    # Daily climatological percentile
    cdo -O -L ydaypctl,95 \
        "$tmp_wet" \
        "$tmp_min" \
        "$tmp_max" \
        "$outfile"

    rm -f \
        "$tmp_jjas" \
        "$tmp_wet" \
        "$tmp_min" \
        "$tmp_max"
}

# ======================================================
# COMPUTE ANNUAL ETCCDI R95pTOT
# ======================================================

compute_r95ptot () {

    infile=$1
    p95=$2
    outfile=$3

    tmp_jjas="$TMP/r95_jjas.nc"

    # JJAS only
    cdo -O -L selmon,6/9 \
        "$infile" \
        "$tmp_jjas"

    years=$(cdo -O -s showyear "$tmp_jjas")

    rm -f "$outfile"

    first=1

    for yy in $years; do

        echo "   -> Year $yy"

        tmp_year="$TMP/r95_year_${yy}.nc"
        tmp_out="$TMP/r95_out_${yy}.nc"
        tmp_final="$TMP/r95_final_${yy}.nc"

        # One year only
        cdo -O -L selyear,$yy \
            "$tmp_jjas" \
            "$tmp_year"

        # ETCCDI R95pTOT
        cdo -O -L eca_r95ptot \
            "$tmp_year" \
            "$p95" \
            "$tmp_out"

        # Rename variable
        cdo -O -L chname,precipitation_percent_due_to_R95p_days,R95pTOT \
            "$tmp_out" \
            "$tmp_final"

        # Merge all years
        if [ $first -eq 1 ]; then

            cp "$tmp_final" "$outfile"

            first=0

        else

            cdo -O -L mergetime \
                "$outfile" \
                "$tmp_final" \
                "${outfile}.tmp"

            mv "${outfile}.tmp" "$outfile"

        fi

    done

    rm -f "$tmp_jjas"
    rm -f "$TMP"/r95_year_*.nc
    rm -f "$TMP"/r95_out_*.nc
    rm -f "$TMP"/r95_final_*.nc
}

# ======================================================
# MAIN FUNCTION
# ======================================================

compute_extremes() {

    TASMAX=$1
    TASMIN=$2
    PR=$3
    SPEI=$4
    OUTFILE=$5
    START=$6
    END=$7

    echo ""
    echo "======================================="
    echo "$OUTFILE"
    echo "======================================="

    rm -f $TMP/*.nc

    # ==================================================
    # VARIABLE NAMES
    # ==================================================

    PR_VAR=$(cdo -O showname "$PR" | head -1 | awk '{print $1}')

    TX_VAR=$(cdo -O showname "$TASMAX" | head -1 | awk '{print $1}')

    TN_VAR=$(cdo -O showname "$TASMIN" | head -1 | awk '{print $1}')

    SPEI_VAR=$(cdo -O showname "$SPEI" | head -1 | awk '{print $1}')

    echo "PR variable   : $PR_VAR"
    echo "TX variable   : $TX_VAR"
    echo "TN variable   : $TN_VAR"
    echo "SPEI variable : $SPEI_VAR"

    # ==================================================
    # SELECT PERIOD
    # ==================================================

    cdo -O seldate,$START,$END \
        "$PR" \
        "$TMP/pr_period.nc"

    cdo -O seldate,$START,$END \
        "$TASMAX" \
        "$TMP/tx_period.nc"

    cdo -O seldate,$START,$END \
        "$TASMIN" \
        "$TMP/tn_period.nc"

    # ==================================================
    # JJAS
    # ==================================================

    cdo -O selmon,6,7,8,9 \
        "$TMP/pr_period.nc" \
        "$TMP/pr_jjas.nc"

    cdo -O selmon,6,7,8,9 \
        "$TMP/tx_period.nc" \
        "$TMP/tx_jjas.nc"

    cdo -O selmon,6,7,8,9 \
        "$TMP/tn_period.nc" \
        "$TMP/tn_jjas.nc"

    # ==================================================
    # REFERENCE PERIOD FOR TEMPERATURE
    # ==================================================

    cdo -O seldate,$REF_START,$REF_END \
        "$TASMAX" \
        "$TMP/ref_tx.nc"

    cdo -O selmon,6,7,8,9 \
        "$TMP/ref_tx.nc" \
        "$TMP/ref_tx_jjas.nc"

    cdo -O seldate,$REF_START,$REF_END \
        "$TASMIN" \
        "$TMP/ref_tn.nc"

    cdo -O selmon,6,7,8,9 \
        "$TMP/ref_tn.nc" \
        "$TMP/ref_tn_jjas.nc"

    # ==================================================
    # REFERENCE PERIOD FOR PRECIP
    # ==================================================

    cdo -O seldate,$REF_START,$REF_END \
        "$PR" \
        "$TMP/ref_pr.nc"

    # ==================================================
    # TXx
    # ==================================================

    cdo -O yearmax \
        "$TMP/tx_jjas.nc" \
        "$TMP/TXx.nc"

    cdo -O chname,$TX_VAR,TXx \
        "$TMP/TXx.nc" \
        "$TMP/TXx_r.nc"

    # ==================================================
    # TNx
    # ==================================================

    cdo -O yearmax \
        "$TMP/tn_jjas.nc" \
        "$TMP/TNx.nc"

    cdo -O chname,$TN_VAR,TNx \
        "$TMP/TNx.nc" \
        "$TMP/TNx_r.nc"

    # ==================================================
    # DTR
    # ==================================================

    cdo -O sub \
        "$TMP/tx_jjas.nc" \
        "$TMP/tn_jjas.nc" \
        "$TMP/dtr_daily.nc"

    cdo -O yearmean \
        "$TMP/dtr_daily.nc" \
        "$TMP/DTR.nc"

    cdo -O chname,$TX_VAR,DTR \
        "$TMP/DTR.nc" \
        "$TMP/DTR_r.nc"

    # ==================================================
    # TX90p
    # ==================================================

    cdo -O timmin \
        "$TMP/ref_tx_jjas.nc" \
        "$TMP/ref_tx_min.nc"

    cdo -O timmax \
        "$TMP/ref_tx_jjas.nc" \
        "$TMP/ref_tx_max.nc"

    cdo -O timpctl,90 \
        "$TMP/ref_tx_jjas.nc" \
        "$TMP/ref_tx_min.nc" \
        "$TMP/ref_tx_max.nc" \
        "$TMP/p90_tx.nc"

    cdo -O gt \
        "$TMP/tx_jjas.nc" \
        "$TMP/p90_tx.nc" \
        "$TMP/tx90_bool.nc"

    cdo -O yearsum \
        "$TMP/tx90_bool.nc" \
        "$TMP/TX90p.nc"

    cdo -O chname,$TX_VAR,TX90p \
        "$TMP/TX90p.nc" \
        "$TMP/TX90p_r.nc"
    # ==================================================
    # TN90p
    # ==================================================

    cdo -O timmin \
    "$TMP/ref_tn_jjas.nc" \
    "$TMP/ref_tn_min.nc"

    cdo -O timmax \
    "$TMP/ref_tn_jjas.nc" \
    "$TMP/ref_tn_max.nc"

    cdo -O timpctl,90 \
    "$TMP/ref_tn_jjas.nc" \
    "$TMP/ref_tn_min.nc" \
    "$TMP/ref_tn_max.nc" \
    "$TMP/p90_tn.nc"

    cdo -O gt \
    "$TMP/tn_jjas.nc" \
    "$TMP/p90_tn.nc" \
    "$TMP/tn90_bool.nc"

    cdo -O yearsum \
    "$TMP/tn90_bool.nc" \
    "$TMP/TN90p.nc"

     cdo -O chname,$TN_VAR,TN90p \
    "$TMP/TN90p.nc" \
    "$TMP/TN90p_r.nc"
   # ==================================================
   # TX35
    # ==================================================

    cdo -O gec,35 \
    "$TMP/tx_jjas.nc" \
    "$TMP/tx35_bool.nc"

    cdo -O yearsum \
    "$TMP/tx35_bool.nc" \
    "$TMP/TX35.nc"

     cdo -O chname,$TX_VAR,TX35 \
    "$TMP/TX35.nc" \
    "$TMP/TX35_r.nc"
    # ==================================================
    # WSDI
    # ==================================================

    cdo -O consecsum \
        "$TMP/tx90_bool.nc" \
        "$TMP/wsdi_consec.nc"

    cdo -O gec,6 \
        "$TMP/wsdi_consec.nc" \
        "$TMP/wsdi_mask.nc"

    cdo -O yearsum \
        "$TMP/wsdi_mask.nc" \
        "$TMP/WSDI.nc"

    cdo -O chname,$TX_VAR,WSDI \
        "$TMP/WSDI.nc" \
        "$TMP/WSDI_r.nc"

    # ==================================================
    # RX1day
    # ==================================================

    cdo -O yearmax \
        "$TMP/pr_jjas.nc" \
        "$TMP/RX1day.nc"

    cdo -O chname,$PR_VAR,RX1day \
        "$TMP/RX1day.nc" \
        "$TMP/RX1day_r.nc"
   # ==================================================
   # RX5day
   # ==================================================

    cdo -O runsum,5 \
    "$TMP/pr_jjas.nc" \
    "$TMP/pr_5day.nc"

    cdo -O yearmax \
    "$TMP/pr_5day.nc" \
    "$TMP/RX5day.nc"

    cdo -O chname,$PR_VAR,RX5day \
    "$TMP/RX5day.nc" \
    "$TMP/RX5day_r.nc"

    # ==================================================
    # CDD
    # ==================================================

    cdo -O ltc,1 \
        "$TMP/pr_jjas.nc" \
        "$TMP/dry.nc"

    cdo -O consecsum \
        "$TMP/dry.nc" \
        "$TMP/consec.nc"

    cdo -O yearmax \
        "$TMP/consec.nc" \
        "$TMP/CDD.nc"

    cdo -O chname,$PR_VAR,CDD \
        "$TMP/CDD.nc" \
        "$TMP/CDD_r.nc"
 # ==================================================
    # CWD
    # ==================================================

    cdo -O gtc,1 \
        "$TMP/pr_jjas.nc" \
        "$TMP/wet.nc"

    cdo -O consecsum \
        "$TMP/wet.nc" \
        "$TMP/consec_wet.nc"

    cdo -O yearmax \
        "$TMP/consec_wet.nc" \
        "$TMP/CWD.nc"

    cdo -O chname,$PR_VAR,CWD \
        "$TMP/CWD.nc" \
        "$TMP/CWD_r.nc"

    # ==================================================
    # PRtot
    # ==================================================

    cdo -O yearsum \
        "$TMP/pr_jjas.nc" \
        "$TMP/PRtot.nc"

    cdo -O chname,$PR_VAR,PRtot \
        "$TMP/PRtot.nc" \
        "$TMP/PRtot_r.nc"

    # ==================================================
    # R95pTOT
    # ==================================================

    compute_p95_reference \
        "$TMP/ref_pr.nc" \
        "$TMP/p95.nc"

    compute_r95ptot \
        "$TMP/pr_period.nc" \
        "$TMP/p95.nc" \
        "$TMP/R95pTOT_r.nc"

    # ==================================================
    # SPEI3
    # ==================================================

    cdo -O seldate,$START,$END \
        "$SPEI" \
        "$TMP/spei_period.nc"

    cdo -O chname,$SPEI_VAR,SPEI3 \
        "$TMP/spei_period.nc" \
        "$TMP/SPEI3.nc"
    # ==================================================
    # SDII
    # ==================================================

    cdo -O setrtomiss,-inf,0.999 \
    "$TMP/pr_jjas.nc" \
    "$TMP/wetdays.nc"

     cdo -O yearsum \
    "$TMP/wetdays.nc" \
    "$TMP/pr_wet_sum.nc"

    cdo -O gec,1 \
    "$TMP/pr_jjas.nc" \
    "$TMP/wet_count.nc"

    cdo -O yearsum \
    "$TMP/wet_count.nc" \
    "$TMP/wet_days.nc"

     cdo -O div \
    "$TMP/pr_wet_sum.nc" \
    "$TMP/wet_days.nc" \
    "$TMP/SDII.nc"

    cdo -O chname,$PR_VAR,SDII \
    "$TMP/SDII.nc" \
    "$TMP/SDII_r.nc"
   # ==================================================
   # R20mm
   # ==================================================

     cdo -O gec,20 \
    "$TMP/pr_jjas.nc" \
    "$TMP/r20_bool.nc"

     cdo -O yearsum \
    "$TMP/r20_bool.nc" \
    "$TMP/R20mm.nc"

     cdo -O chname,$PR_VAR,R20mm \
    "$TMP/R20mm.nc" \
    "$TMP/R20mm_r.nc"
       # ==================================================
   # R10mm
   # ==================================================

     cdo -O gec,10 \
    "$TMP/pr_jjas.nc" \
    "$TMP/r10_bool.nc"

     cdo -O yearsum \
    "$TMP/r10_bool.nc" \
    "$TMP/R10mm.nc"

     cdo -O chname,$PR_VAR,R10mm \
    "$TMP/R10mm.nc" \
    "$TMP/R10mm_r.nc"
         # ==================================================
   # R10mm
   # ==================================================

     cdo -O gec,1 \
    "$TMP/pr_jjas.nc" \
    "$TMP/r1_bool.nc"

     cdo -O yearsum \
    "$TMP/r1_bool.nc" \
    "$TMP/R1mm.nc"

     cdo -O chname,$PR_VAR,R1mm \
    "$TMP/R1mm.nc" \
    "$TMP/R1mm_r.nc"

    # ==================================================
    # REMOVE OLD FILE
    # ==================================================

    [ -f "$OUTFILE" ] && rm -f "$OUTFILE"

    # ==================================================
    # MERGE
    # ==================================================

    cdo -O merge \
        "$TMP/TXx_r.nc" \
        "$TMP/TNx_r.nc" \
        "$TMP/DTR_r.nc" \
        "$TMP/TX90p_r.nc" \
        "$TMP/WSDI_r.nc" \
        "$TMP/RX1day_r.nc" \
        "$TMP/RX5day_r.nc" \
        "$TMP/CDD_r.nc" \
        "$TMP/CWD_r.nc" \
        "$TMP/PRtot_r.nc" \
        "$TMP/R95pTOT_r.nc" \
        "$TMP/SPEI3.nc" \
        "$TMP/SDII_r.nc"\
        "$TMP/R1mm_r.nc"\
        "$TMP/R10mm_r.nc"\
        "$TMP/R20mm_r.nc"\
        "$TMP/TN90p_r.nc"\
        "$TMP/TX35_r.nc"\
        "$OUTFILE"

    echo ""
    echo "DONE"
}

# ======================================================
# OBSERVATIONS
# ======================================================

OBS_PR="$OBS_DIR/ARC2_merged.nc"

OBS_TX="$OBS_DIR/CHIRTS_Tmax_merged.nc"

OBS_TN="$OBS_DIR/CHIRTS_Tmin_merged.nc"

OBS_SPEI="$SPEI_DIR/observations_spei3.nc"

compute_extremes \
    "$OBS_TX" \
    "$OBS_TN" \
    "$OBS_PR" \
    "$OBS_SPEI" \
    "$OUT/observations/observations_extremes.nc" \
    "$REF_START" \
    "$REF_END"

# ======================================================
# CMIP6
# ======================================================

declare -A DONE_HIST

for SCENARIO in ssp245 ssp585
do

    echo ""
    echo "SCENARIO $SCENARIO"

    for PR in $CMIP6_DIR/pr/*${SCENARIO}*.nc
    do

        FILE=$(basename "$PR")

        MODEL=$(echo "$FILE" | cut -d "_" -f1)

        echo ""
        echo "$MODEL"

        TX="$CMIP6_DIR/tasmax/${FILE/prAdjust/tasmaxAdjust}"
        TX="${TX/_pr/_tasmax}"

        TN="$CMIP6_DIR/tasmin/${FILE/prAdjust/tasminAdjust}"
        TN="${TN/_pr/_tasmin}"

        [ -f "$TX" ] || continue
        [ -f "$TN" ] || continue

        # ==============================================
        # SPEI FILES
        # ==============================================

        SPEI_HIST="$SPEI_DIR/historical/${MODEL}_historical_spei3.nc"

        SPEI_NEAR="$SPEI_DIR/$SCENARIO/${MODEL}_${SCENARIO}_near_spei3.nc"

        SPEI_FAR="$SPEI_DIR/$SCENARIO/${MODEL}_${SCENARIO}_far_spei3.nc"

        # ==============================================
        # HISTORICAL
        # ==============================================

        if [ -z "${DONE_HIST[$MODEL]}" ]; then

            [ -f "$SPEI_HIST" ] || continue

            compute_extremes \
                "$TX" \
                "$TN" \
                "$PR" \
                "$SPEI_HIST" \
                "$OUT/historical/${MODEL}_historical_extremes.nc" \
                "$REF_START" \
                "$REF_END"

            DONE_HIST[$MODEL]=1

        fi

        # ==============================================
        # NEAR FUTURE
        # ==============================================

        [ -f "$SPEI_NEAR" ] || continue

        compute_extremes \
            "$TX" \
            "$TN" \
            "$PR" \
            "$SPEI_NEAR" \
            "$OUT/$SCENARIO/${MODEL}_${SCENARIO}_near_extremes.nc" \
            "$NEAR_START" \
            "$NEAR_END"

        # ==============================================
        # FAR FUTURE
        # ==============================================

        [ -f "$SPEI_FAR" ] || continue

        compute_extremes \
            "$TX" \
            "$TN" \
            "$PR" \
            "$SPEI_FAR" \
            "$OUT/$SCENARIO/${MODEL}_${SCENARIO}_far_extremes.nc" \
            "$FAR_START" \
            "$FAR_END"

    done

done

echo ""
echo "--------------------------------"
echo "FINISHED"
echo "--------------------------------"
