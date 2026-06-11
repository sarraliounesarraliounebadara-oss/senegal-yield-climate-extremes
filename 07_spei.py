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

print("\nDONE")import xarray as xr
import os
import numpy as np
from scipy.stats import gamma, norm

BASE = "/media/sarr/01DC5DE9D15E8CF0/climate_data_in_mask"
OUTBASE = "/media/sarr/01DC5DE9D15E8CF0/climate_extremes"

LOGFILE = "spei_errors.log"


CROPS = ["maize","millet","sorghum","groundnut"]

PERIODS = ["historical","near_future","far_future"]

MODELS = [
"ACCESS-CM2","ACCESS-ESM1-5","BCC-CSM2-MR","CNRM-CM6-1",
"FGOALS-g3","GFDL-ESM4","MIROC6","MPI-ESM1-2-LR",
"MRI-ESM2-0","CCCma"
]

SCENARIOS = ["ssp245","ssp585"]

DEPARTMENTS = [
"bakel","bambey","bignona","diourbel","fatick","foundiougne",
"gossas","kaffrine","kaolack","kolda","kébémer","kédougou",
"linguère","louga","mbacké","mbour","nioro_du_rip",
"oussouye","sédhiou","tambacounda","thiès","tivaouane",
"vélingara","ziguinchor"
]

################################
# LOG
################################
def log_error(msg):
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")

################################
# PET
################################
def hargreaves_pet(tmin, tmax):
    tmean = (tmin + tmax) / 2
    return 0.0023 * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0))

################################
# detect variable
################################
def find_var(ds, keywords):
    for v in ds.data_vars:
        for k in keywords:
            if k in v.lower():
                return v
    return list(ds.data_vars)[0]

################################
# SPEI
################################
def compute_spei_manual(pr, pet, pr_hist, pet_hist, scale=3):

    wb = pr - pet
    wb_hist = pr_hist - pet_hist

    if wb.time.size < scale:
        raise ValueError("Not enough time steps")

    wb_roll = wb.rolling(time=scale).sum()
    wb_hist_roll = wb_hist.rolling(time=scale).sum()

    data = wb_hist_roll.values.flatten()
    data = data[~np.isnan(data)]

    if len(data) < 20:
        raise ValueError("Not enough data for gamma fit")

    shape, loc, scale_param = gamma.fit(data)

    cdf = gamma.cdf(wb_roll, shape, loc=loc, scale=scale_param)
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)

    spei = norm.ppf(cdf)

    return xr.DataArray(
        spei,
        coords=wb_roll.coords,
        dims=wb_roll.dims,
        name="SPEI"
    )

################################
# CALCUL SPEI
################################
def compute_spei(pr_file, tmin_file, tmax_file,
                 pr_hist, tmin_hist, tmax_hist,
                 out_file):

    try:
        print("➡️", os.path.basename(pr_file))

        if os.path.exists(out_file):
            os.remove(out_file)

        pr = xr.open_dataset(pr_file)
        tmin = xr.open_dataset(tmin_file)
        tmax = xr.open_dataset(tmax_file)

        pr_h = xr.open_dataset(pr_hist)
        tmin_h = xr.open_dataset(tmin_hist)
        tmax_h = xr.open_dataset(tmax_hist)

        pr_var = find_var(pr, ["pr","precip"])
        tmin_var = find_var(tmin, ["tmin"])
        tmax_var = find_var(tmax, ["tmax"])

        ################################
        # RESAMPLE
        ################################
        pr_data = pr[pr_var].sortby("time").resample(time="MS").sum()
        tmin_data = tmin[tmin_var].sortby("time").resample(time="MS").mean()
        tmax_data = tmax[tmax_var].sortby("time").resample(time="MS").mean()

        pr_hist_data = pr_h[pr_var].sortby("time").resample(time="MS").sum()
        tmin_hist_data = tmin_h[tmin_var].sortby("time").resample(time="MS").mean()
        tmax_hist_data = tmax_h[tmax_var].sortby("time").resample(time="MS").mean()

        ################################
        # ALIGN
        ################################
        pr_data, tmin_data, tmax_data = xr.align(pr_data, tmin_data, tmax_data, join="inner")
        pr_hist_data, tmin_hist_data, tmax_hist_data = xr.align(
            pr_hist_data, tmin_hist_data, tmax_hist_data, join="inner"
        )

        ################################
        # SPEI
        ################################
        pet = hargreaves_pet(tmin_data, tmax_data)
        pet_hist = hargreaves_pet(tmin_hist_data, tmax_hist_data)

        spei = compute_spei_manual(pr_data, pet,
                                  pr_hist_data, pet_hist)

        ################################
        # JJAS
        ################################
        spei = spei.sel(time=spei["time.month"].isin([6,7,8,9]))

        # moyenne saisonnière (cohérent avec ton pipeline)
        spei = spei.groupby("time.year").mean("time")

        ################################
        # SAVE
        ################################
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        spei.to_netcdf(out_file)

        print("saved")

        pr.close(); tmin.close(); tmax.close()
        pr_h.close(); tmin_h.close(); tmax_h.close()

    except Exception as e:
        msg = f"ERROR | {pr_file} | {str(e)}"
        print(msg)
        log_error(msg)

################################
# MAIN
################################
for crop in CROPS:

    print("\n==============================")
    print("CROP:", crop)
    print("==============================")

    ################################
    # OBS
    ################################
    print("OBS")

    for dep in DEPARTMENTS:

        pr_file = f"{BASE}/{crop}/observation/chirps/chirps_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"
        tmin_file = f"{BASE}/{crop}/observation/ERA5/era5_tmin_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"
        tmax_file = f"{BASE}/{crop}/observation/ERA5/era5_tmax_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"

        if not os.path.exists(pr_file):
            continue

        if not os.path.exists(tmin_file) or not os.path.exists(tmax_file):
            log_error(f"MISSING OBS TEMP | {dep}")
            continue

        out_file = f"{OUTBASE}/{crop}/observation/chirps_{crop}_historical_mask_{crop}_{dep}_4km_fldmean_SPEI3.nc"

        compute_spei(pr_file, tmin_file, tmax_file,
                     pr_file, tmin_file, tmax_file,
                     out_file)

    ################################
    # CMIP6
    ################################
    print("CMIP6")

    BASECROP = f"{BASE}/{crop}/cmip6"

    for period in PERIODS:

        print("➡️", period)

        for model in MODELS:

            if period == "historical":
                scenarios_loop = [None]
            else:
                scenarios_loop = SCENARIOS

            for scen in scenarios_loop:
                for dep in DEPARTMENTS:

                    if period == "historical":

                        pr_file = f"{BASECROP}/historical/pr_{model}_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"
                        tmin_file = f"{BASECROP}/historical/tmin_{model}_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"
                        tmax_file = f"{BASECROP}/historical/tmax_{model}_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"

                        pr_hist = pr_file
                        tmin_hist = tmin_file
                        tmax_hist = tmax_file

                    else:

                        pr_file = f"{BASECROP}/{period}/pr_{model}_{scen}_{crop}_{period}_mask_{crop}_{dep}_4km_fldmean.nc"
                        tmin_file = f"{BASECROP}/{period}/tmin_{model}_{scen}_{crop}_{period}_mask_{crop}_{dep}_4km_fldmean.nc"
                        tmax_file = f"{BASECROP}/{period}/tmax_{model}_{scen}_{crop}_{period}_mask_{crop}_{dep}_4km_fldmean.nc"

                        pr_hist = f"{BASECROP}/historical/pr_{model}_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"
                        tmin_hist = f"{BASECROP}/historical/tmin_{model}_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"
                        tmax_hist = f"{BASECROP}/historical/tmax_{model}_{crop}_historical_mask_{crop}_{dep}_4km_fldmean.nc"

                    if not os.path.exists(pr_file):
                        continue

                    if not os.path.exists(tmin_file) or not os.path.exists(tmax_file):
                        log_error(f"MISSING TEMP | {model} {scen} {dep}")
                        continue

                    if not os.path.exists(pr_hist) \
                    or not os.path.exists(tmin_hist) \
                    or not os.path.exists(tmax_hist):

                        log_error(f"MISSING HIST FULL | {model} {dep}")
                        continue

                    if period == "historical":
                        out_file = f"{OUTBASE}/{crop}/cmip6/historical/pr_{model}_{crop}_historical_mask_{crop}_{dep}_4km_fldmean_SPEI3.nc"
                    else:
                        out_file = f"{OUTBASE}/{crop}/cmip6/{period}/pr_{model}_{scen}_{crop}_{period}_mask_{crop}_{dep}_4km_fldmean_SPEI3.nc"

                    compute_spei(pr_file, tmin_file, tmax_file,
                                 pr_hist, tmin_hist, tmax_hist,
                                 out_file)

print("\nSPEI-3 calculation finished")
