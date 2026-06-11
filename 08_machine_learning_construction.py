#!/usr/bin/env python3

import xarray as xr
import numpy as np
import os

# =========================================================
# PATHS
# =========================================================

BASE = "/media/sarr/01DC5DE9D15E8CF0/departments/climate_data"

OBS_DIR = f"{BASE}/anacim_data"

CMIP6_DIR = f"{BASE}/cmip6"

OUT_DIR = f"{BASE}/spei3"

HIST_DIR = f"{OUT_DIR}/historical"

SSP245_DIR = f"{OUT_DIR}/ssp245"

SSP585_DIR = f"{OUT_DIR}/ssp585"

os.makedirs(OUT_DIR, exist_ok=True)

os.makedirs(HIST_DIR, exist_ok=True)

os.makedirs(SSP245_DIR, exist_ok=True)

os.makedirs(SSP585_DIR, exist_ok=True)

# =========================================================
# PERIODS
# =========================================================

HIST_START = "1984-01-01"
HIST_END   = "2013-12-31"

NEAR_START = "2036-01-01"
NEAR_END   = "2065-12-31"

FAR_START  = "2071-01-01"
FAR_END    = "2100-12-31"

JJAS = [6, 7, 8, 9]

# =========================================================
# UTILITIES
# =========================================================

def get_var(ds):

    return list(ds.data_vars)[0]

# =========================================================
# EXTRATERRESTRIAL RADIATION
# =========================================================

def compute_ra(lat, time):

    Gsc = 0.0820

    doy = xr.DataArray(
        time.dt.dayofyear,
        dims=["time"],
        coords={"time": time}
    )

    phi = np.deg2rad(lat)

    dr = 1 + 0.033 * np.cos(
        2 * np.pi * doy / 365
    )

    delta = 0.409 * np.sin(
        (2 * np.pi * doy / 365) - 1.39
    )

    ws = np.arccos(
        -np.tan(phi) * np.tan(delta)
    )

    ra = (
        (24 * 60 / np.pi)
        * Gsc
        * dr
        * (
            ws * np.sin(phi) * np.sin(delta)
            + np.cos(phi)
            * np.cos(delta)
            * np.sin(ws)
        )
    )

    return ra

# =========================================================
# PET HARGREAVES
# =========================================================

def compute_pet(tmin, tmax):

    tmean = (tmin + tmax) / 2

    lat_name = [
        d for d in tmin.dims
        if "lat" in d.lower()
    ][0]

    lat = tmin[lat_name]

    ra = compute_ra(
        lat,
        tmin.time
    )

    ra = ra.broadcast_like(tmin)

    # =====================================================
    # HARGREAVES PET (mm/day)
    # =====================================================

    pet_daily = (
        0.0023
        * ra
        * (tmean + 17.8)
        * np.sqrt(
            np.maximum(
                tmax - tmin,
                0
            )
        )
    )

    pet_daily = pet_daily.clip(min=0)

    # =====================================================
    # CONVERT TO MONTHLY TOTAL
    # =====================================================

    days_in_month = xr.DataArray(
        tmin.time.dt.days_in_month,
        dims=["time"],
        coords={"time": tmin.time}
    )

    pet = pet_daily * days_in_month

    pet.name = "PET"

    return pet

# =========================================================
# SPEI3
# =========================================================

def compute_spei(
    pr,
    pet,
    pr_hist,
    pet_hist
):

    # =====================================================
    # WATER BALANCE
    # =====================================================

    wb = pr - pet

    wb_hist = pr_hist - pet_hist

    # =====================================================
    # 3-MONTH ACCUMULATION
    # =====================================================

    wb3 = wb.rolling(
        time=3,
        min_periods=3
    ).sum()

    wb3_hist = wb_hist.rolling(
        time=3,
        min_periods=3
    ).sum()

    # =====================================================
    # MONTHLY CLIMATOLOGY
    # =====================================================

    clim_mean = wb3_hist.groupby(
        "time.month"
    ).mean("time")

    clim_std = wb3_hist.groupby(
        "time.month"
    ).std("time")

    # Avoid division by zero
    clim_std = xr.where(
        clim_std < 1e-6,
        1e-6,
        clim_std
    )

    # =====================================================
    # STANDARDIZATION
    # =====================================================

    spei = (
        wb3.groupby("time.month")
        - clim_mean
    ) / clim_std

    spei.name = "SPEI3"

    # =====================================================
    # DIAGNOSTICS
    # =====================================================

    print("")
    print("SPEI statistics")
    print("Min  :", float(spei.min()))
    print("Max  :", float(spei.max()))
    print("Mean :", float(spei.mean()))
    print("")

    return spei

# =========================================================
# MAIN
# =========================================================

def compute_spei3(
    pr_file,
    tmin_file,
    tmax_file,
    pr_hist_file,
    tmin_hist_file,
    tmax_hist_file,
    out_file,
    start,
    end
):

    print("\n===================================")

    print(out_file)

    print("===================================")

    # =====================================================
    # OPEN
    # =====================================================

    ds_pr = xr.open_dataset(pr_file)

    ds_tmin = xr.open_dataset(tmin_file)

    ds_tmax = xr.open_dataset(tmax_file)

    ds_pr_hist = xr.open_dataset(pr_hist_file)

    ds_tmin_hist = xr.open_dataset(tmin_hist_file)

    ds_tmax_hist = xr.open_dataset(tmax_hist_file)

    # =====================================================
    # VARIABLE NAMES
    # =====================================================

    pr_var = get_var(ds_pr)

    tmin_var = get_var(ds_tmin)

    tmax_var = get_var(ds_tmax)

    # =====================================================
    # SELECT PERIOD
    # =====================================================

    pr = ds_pr[pr_var].sel(
        time=slice(start, end)
    )

    tmin = ds_tmin[tmin_var].sel(
        time=slice(start, end)
    )

    tmax = ds_tmax[tmax_var].sel(
        time=slice(start, end)
    )

    # =====================================================
    # CONVERT PRECIP TO mm/day IF NEEDED
    # =====================================================

    if float(pr.max()) < 1:

        print("Converting precipitation from kg m-2 s-1 to mm/day")

        pr = pr * 86400

    # =====================================================
    # MONTHLY
    # =====================================================

    pr_month = pr.resample(
        time="MS"
    ).sum()

    tmin_month = tmin.resample(
        time="MS"
    ).mean()

    tmax_month = tmax.resample(
        time="MS"
    ).mean()

    # =====================================================
    # HISTORICAL REFERENCE
    # =====================================================

    pr_hist = (
        ds_pr_hist[pr_var]
        .sel(
            time=slice(
                HIST_START,
                HIST_END
            )
        )
    )

    # =====================================================
    # CONVERT HIST PRECIP
    # =====================================================

    if float(pr_hist.max()) < 1:

        pr_hist = pr_hist * 86400

    pr_hist = pr_hist.resample(
        time="MS"
    ).sum()

    tmin_hist = (
        ds_tmin_hist[tmin_var]
        .sel(
            time=slice(
                HIST_START,
                HIST_END
            )
        )
        .resample(time="MS")
        .mean()
    )

    tmax_hist = (
        ds_tmax_hist[tmax_var]
        .sel(
            time=slice(
                HIST_START,
                HIST_END
            )
        )
        .resample(time="MS")
        .mean()
    )

    # =====================================================
    # ALIGN
    # =====================================================

    pr_month, tmin_month, tmax_month = xr.align(
        pr_month,
        tmin_month,
        tmax_month,
        join="inner"
    )

    pr_hist, tmin_hist, tmax_hist = xr.align(
        pr_hist,
        tmin_hist,
        tmax_hist,
        join="inner"
    )

    # =====================================================
    # PET
    # =====================================================

    pet = compute_pet(
        tmin_month,
        tmax_month
    )

    pet_hist = compute_pet(
        tmin_hist,
        tmax_hist
    )

    # =====================================================
    # SPEI3
    # =====================================================

    spei3 = compute_spei(
        pr_month,
        pet,
        pr_hist,
        pet_hist
    )

    # =====================================================
    # JJAS ONLY
    # =====================================================

    spei3 = spei3.sel(
        time=spei3.time.dt.month.isin(JJAS)
    )

    # =====================================================
    # SEASONAL DRYNESS SIGNAL
    # =====================================================

    spei3 = spei3.groupby(
        "time.year"
    ).min("time")

    # =====================================================
    # CONVERT YEAR -> TIME
    # =====================================================

    years = spei3.year.values

    times = np.array([
        np.datetime64(
            f"{int(y)}-08-01"
        )
        for y in years
    ])

    spei3 = spei3.rename(
        {"year": "time"}
    )

    spei3 = spei3.assign_coords(
        time=times
    )

    spei3.name = "SPEI3"

    # =====================================================
    # SAVE
    # =====================================================

    os.makedirs(
        os.path.dirname(out_file),
        exist_ok=True
    )

    spei3.to_netcdf(out_file)

    print("saved")

# =========================================================
# OBSERVATIONS
# =========================================================

OBS_PR = (
    f"{OBS_DIR}/ARC2_merged.nc"
)

OBS_TMIN = (
    f"{OBS_DIR}/CHIRTS_Tmin_merged.nc"
)

OBS_TMAX = (
    f"{OBS_DIR}/CHIRTS_Tmax_merged.nc"
)

compute_spei3(
    OBS_PR,
    OBS_TMIN,
    OBS_TMAX,
    OBS_PR,
    OBS_TMIN,
    OBS_TMAX,
    f"{OUT_DIR}/observations_spei3.nc",
    HIST_START,
    HIST_END
)

# =========================================================
# CMIP6
# =========================================================

done_hist = set()

for scenario in ["ssp245", "ssp585"]:

    print("\n==============================")

    print(scenario)

    print("==============================")

    for pr_file in sorted(
        os.listdir(
            f"{CMIP6_DIR}/pr"
        )
    ):

        if scenario not in pr_file:
            continue

        model = pr_file.split("_")[0]

        print("\n", model)

        pr_path = (
            f"{CMIP6_DIR}/pr/{pr_file}"
        )

        tmin_file = (
            pr_file
            .replace(
                "prAdjust",
                "tasminAdjust"
            )
            .replace(
                "_pr_",
                "_tasmin_"
            )
        )

        tmax_file = (
            pr_file
            .replace(
                "prAdjust",
                "tasmaxAdjust"
            )
            .replace(
                "_pr_",
                "_tasmax_"
            )
        )

        tmin_path = (
            f"{CMIP6_DIR}/tasmin/{tmin_file}"
        )

        tmax_path = (
            f"{CMIP6_DIR}/tasmax/{tmax_file}"
        )

        if not os.path.exists(tmin_path):
            continue

        if not os.path.exists(tmax_path):
            continue

        # =================================================
        # HISTORICAL ONLY ONCE
        # =================================================

        if model not in done_hist:

            compute_spei3(
                pr_path,
                tmin_path,
                tmax_path,
                pr_path,
                tmin_path,
                tmax_path,
                (
                    f"{HIST_DIR}/"
                    f"{model}_historical_spei3.nc"
                ),
                HIST_START,
                HIST_END
            )

            done_hist.add(model)

        # =================================================
        # NEAR
        # =================================================

        compute_spei3(
            pr_path,
            tmin_path,
            tmax_path,
            pr_path,
            tmin_path,
            tmax_path,
            (
                f"{OUT_DIR}/"
                f"{scenario}/"
                f"{model}_{scenario}"
                f"_near_spei3.nc"
            ),
            NEAR_START,
            NEAR_END
        )

        # =================================================
        # FAR
        # =================================================

        compute_spei3(
            pr_path,
            tmin_path,
            tmax_path,
            pr_path,
            tmin_path,
            tmax_path,
            (
                f"{OUT_DIR}/"
                f"{scenario}/"
                f"{model}_{scenario}"
                f"_far_spei3.nc"
            ),
            FAR_START,
            FAR_END
        )

print("\n===================================")

print("SPEI3 FINISHED")

print("===================================")
