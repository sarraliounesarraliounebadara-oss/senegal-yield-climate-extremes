#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPEI3-Makkink CMIP6 - VERSION FAST CORRIGÉE

Correction importante :
la standardisation mensuelle est appliquée mois par mois sans créer
une dimension supplémentaire "month".

Sortie attendue pour chaque fichier :
  SPEI3(time, lat, lon)
avec une seule valeur par pixel et par année.
"""

import os
import warnings
import numpy as np
import xarray as xr
import dask

# Lecture/calcul Dask séquentiels pour éviter les erreurs NetCDF/HDF
# observées avec ACCESS-ESM1-5 en lecture parallèle.
dask.config.set(scheduler="single-threaded")

ROOT = "/mnt/disque/departments/climate_data/cmip6"

DIR_PR = os.path.join(ROOT, "pr")
DIR_TMIN = os.path.join(ROOT, "tasmin")
DIR_TMAX = os.path.join(ROOT, "tasmax")
DIR_SRAD = os.path.join(ROOT, "srad")

# Même chemin de sortie que le script FAST précédent
OUT_DIR = os.path.join(
    ROOT,
    "spei3_makkink_fast"
)

HIST_DIR = os.path.join(
    OUT_DIR,
    "historical"
)

SSP245_DIR = os.path.join(
    OUT_DIR,
    "ssp245"
)

SSP585_DIR = os.path.join(
    OUT_DIR,
    "ssp585"
)

for d in [
    OUT_DIR,
    HIST_DIR,
    SSP245_DIR,
    SSP585_DIR
]:
    os.makedirs(
        d,
        exist_ok=True
    )

HIST_START = "1984-01-01"
HIST_END   = "2013-12-31"

NEAR_START = "2036-01-01"
NEAR_END   = "2065-12-31"

FAR_START  = "2071-01-01"
FAR_END    = "2100-12-31"

INPUT_MONTHS = [4, 5, 6, 7, 8, 9]
JJAS = [6, 7, 8, 9]

MAKKINK_A = 0.61
MAKKINK_B = -0.12
PRESSURE_KPA = 101.3
GAMMA_PSY = 0.000665 * PRESSURE_KPA

CHUNKS = {
    "time": 366,
    "lat": 41,
    "lon": 57,
}


def get_paths(model, scenario):

    return {

        "pr": os.path.join(
            DIR_PR,
            f"{model}_{scenario}_prAdjust_CDFt-L-1V-0L.nc"
        ),

        "tmin": os.path.join(
            DIR_TMIN,
            f"{model}_{scenario}_tasminAdjust_CDFt-L-1V-0L.nc"
        ),

        "tmax": os.path.join(
            DIR_TMAX,
            f"{model}_{scenario}_tasmaxAdjust_CDFt-L-1V-0L.nc"
        ),

        "srad": os.path.join(
            DIR_SRAD,
            f"{model}_{scenario}_sradAdjust_CDFt-L-1V-0L.nc"
        ),
    }


def open_model(model, scenario):

    paths = get_paths(
        model,
        scenario
    )

    for path in paths.values():

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                path
            )

    ds_pr = xr.open_dataset(
        paths["pr"],
        chunks=CHUNKS,
        decode_times=True,
        engine="netcdf4"
    )

    ds_tmin = xr.open_dataset(
        paths["tmin"],
        chunks=CHUNKS,
        decode_times=True,
        engine="netcdf4"
    )

    ds_tmax = xr.open_dataset(
        paths["tmax"],
        chunks=CHUNKS,
        decode_times=True,
        engine="netcdf4"
    )

    ds_srad = xr.open_dataset(
        paths["srad"],
        chunks=CHUNKS,
        decode_times=True,
        engine="netcdf4"
    )

    pr = ds_pr["prAdjust"]
    tmin = ds_tmin["tasminAdjust"]
    tmax = ds_tmax["tasmaxAdjust"]
    srad = ds_srad["sradAdjust"]

    for name, da in [
        ("tmin", tmin),
        ("tmax", tmax)
    ]:

        units = str(
            da.attrs.get(
                "units",
                ""
            )
        ).strip().lower()

        if units in {
            "k",
            "kelvin"
        }:

            if name == "tmin":
                tmin = da - 273.15
            else:
                tmax = da - 273.15

    print(
        "      pr units   :",
        pr.attrs.get(
            "units",
            "?"
        )
    )

    print(
        "      Tmin units :",
        tmin.attrs.get(
            "units",
            "?"
        )
    )

    print(
        "      Tmax units :",
        tmax.attrs.get(
            "units",
            "?"
        )
    )

    print(
        "      srad units :",
        srad.attrs.get(
            "units",
            "?"
        )
    )

    return [
        ds_pr,
        ds_tmin,
        ds_tmax,
        ds_srad
    ], pr, tmin, tmax, srad


def close_all(datasets):

    for ds in datasets:

        ds.close()


def pet_makkink_monthly(
    tmin_month,
    tmax_month,
    srad_month
):

    tmean = (
        tmin_month
        + tmax_month
    ) / 2.0

    es = 0.6108 * np.exp(
        (17.27 * tmean)
        / (tmean + 237.3)
    )

    delta = (
        4098.0
        * es
        / ((tmean + 237.3) ** 2)
    )

    lambda_v = (
        2.501
        - 0.002361 * tmean
    )

    pet_daily = (
        MAKKINK_A
        * (
            delta
            / (
                delta
                + GAMMA_PSY
            )
        )
        * (
            srad_month
            / lambda_v
        )
        + MAKKINK_B
    )

    pet_daily = pet_daily.clip(
        min=0.0
    )

    days = xr.DataArray(
        tmin_month.time.dt.days_in_month,
        dims=[
            "time"
        ],
        coords={
            "time":
            tmin_month.time
        }
    )

    return (
        pet_daily
        * days
    )


def select_april_september(
    da,
    start,
    end
):

    da = da.sel(
        time=slice(
            start,
            end
        )
    )

    return da.where(
        da.time.dt.month.isin(
            INPUT_MONTHS
        ),
        drop=True
    )


def compute_wb3_jjas(
    pr,
    tmin,
    tmax,
    srad,
    start,
    end
):

    pr = select_april_september(
        pr,
        start,
        end
    )

    tmin = select_april_september(
        tmin,
        start,
        end
    )

    tmax = select_april_september(
        tmax,
        start,
        end
    )

    srad = select_april_september(
        srad,
        start,
        end
    )

    pr, tmin, tmax, srad = xr.align(
        pr,
        tmin,
        tmax,
        srad,
        join="inner"
    )

    print(
        "      jours retenus:",
        pr.sizes["time"]
    )

    p_month = pr.resample(
        time="MS"
    ).sum(
        skipna=False
    )

    tmin_month = tmin.resample(
        time="MS"
    ).mean(
        skipna=False
    )

    tmax_month = tmax.resample(
        time="MS"
    ).mean(
        skipna=False
    )

    srad_month = srad.resample(
        time="MS"
    ).mean(
        skipna=False
    )

    (
        p_month,
        tmin_month,
        tmax_month,
        srad_month
    ) = xr.align(
        p_month,
        tmin_month,
        tmax_month,
        srad_month,
        join="inner"
    )

    pet_month = pet_makkink_monthly(
        tmin_month,
        tmax_month,
        srad_month
    )

    wb = (
        p_month
        - pet_month
    )

    wb3 = wb.rolling(
        time=3,
        min_periods=3
    ).sum()

    wb3 = wb3.where(
        wb3.time.dt.month.isin(
            JJAS
        ),
        drop=True
    )

    print(
        "      calcul bilan 3 mois..."
    )

    wb3 = wb3.astype(
        np.float32
    ).compute(
        scheduler="single-threaded"
    )

    print(
        "      bilan calculé:",
        wb3.sizes["time"],
        "mois"
    )

    return wb3


def historical_climatology(
    wb3_hist
):

    clim_mean = wb3_hist.groupby(
        "time.month"
    ).mean(
        "time"
    )

    clim_std = wb3_hist.groupby(
        "time.month"
    ).std(
        "time"
    )

    clim_std = xr.where(
        clim_std < 1.0e-6,
        1.0e-6,
        clim_std
    )

    return (
        clim_mean,
        clim_std
    )


def standardized_spei(
    wb3,
    clim_mean,
    clim_std
):

    """
    IMPORTANT :
    on refait groupby("time.month") pour la division.
    Cela empêche la création d'une dimension supplémentaire 'month'.
    """

    anomaly = (
        wb3.groupby(
            "time.month"
        )
        - clim_mean
    )

    spei = (
        anomaly.groupby(
            "time.month"
        )
        / clim_std
    )

    if "month" in spei.dims:

        raise RuntimeError(
            "ERREUR: dimension supplémentaire "
            "'month' dans SPEI mensuel."
        )

    return spei


def annual_min_spei(
    wb3,
    clim_mean,
    clim_std
):

    spei_monthly = standardized_spei(
        wb3,
        clim_mean,
        clim_std
    )

    spei_annual = spei_monthly.groupby(
        "time.year"
    ).min(
        "time",
        skipna=False
    )

    if "month" in spei_annual.dims:

        raise RuntimeError(
            "ERREUR: la sortie annuelle "
            "contient encore une dimension 'month'."
        )

    years = (
        spei_annual.year.values
    )

    times = np.array([
        np.datetime64(
            f"{int(y)}-08-01"
        )
        for y in years
    ])

    spei_annual = spei_annual.rename(
        {
            "year":
            "time"
        }
    )

    spei_annual = spei_annual.assign_coords(
        time=times
    )

    spei_annual.name = "SPEI3"

    spei_annual.attrs = {

        "long_name":
        "Minimum JJAS SPEI3 based on Makkink PET",

        "units":
        "1",

        "PET_method":
        "Makkink",

        "SPEI_scale":
        "3 months",

        "season":
        "JJAS",

        "seasonal_aggregation":
        "minimum monthly SPEI3 during JJAS",

        "reference_period":
        "1984-2013",

        "standardization":
        (
            "monthly historical mean and standard deviation "
            "of 3-month climatic water balance"
        ),

        "radiation_variable":
        "sradAdjust",
    }

    return spei_annual.astype(
        np.float32
    )


def save_spei(
    spei,
    outfile
):

    print(
        "      dimensions finales:",
        spei.dims
    )

    print(
        "      sizes:",
        dict(
            spei.sizes
        )
    )

    spei.to_netcdf(
        outfile,
        encoding={
            "SPEI3": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32"
            }
        }
    )

    print(
        "      min/mean/max:",
        f"{float(spei.min(skipna=True)):.3f}",
        f"{float(spei.mean(skipna=True)):.3f}",
        f"{float(spei.max(skipna=True)):.3f}"
    )

    print(
        "      saved:",
        outfile
    )


def process_model(
    model
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MODEL =",
        model
    )

    print(
        "=" * 80
    )

    # ---------------------------------------------
    # HISTORICAL
    # ---------------------------------------------

    print(
        "\n[1/5] historical 1984-2013"
    )

    (
        datasets,
        pr,
        tmin,
        tmax,
        srad
    ) = open_model(
        model,
        "ssp245"
    )

    try:

        wb3_hist = compute_wb3_jjas(
            pr,
            tmin,
            tmax,
            srad,
            HIST_START,
            HIST_END
        )

    finally:

        close_all(
            datasets
        )

    clim_mean, clim_std = historical_climatology(
        wb3_hist
    )

    spei_hist = annual_min_spei(
        wb3_hist,
        clim_mean,
        clim_std
    )

    hist_file = os.path.join(
        HIST_DIR,
        f"{model}_historical_spei3_makkink_1984-2013.nc"
    )

    save_spei(
        spei_hist,
        hist_file
    )

    # ---------------------------------------------
    # FUTURES
    # ---------------------------------------------

    periods = [

        (
            "ssp245",
            "near",
            NEAR_START,
            NEAR_END
        ),

        (
            "ssp245",
            "far",
            FAR_START,
            FAR_END
        ),

        (
            "ssp585",
            "near",
            NEAR_START,
            NEAR_END
        ),

        (
            "ssp585",
            "far",
            FAR_START,
            FAR_END
        ),
    ]

    for i, (
        scenario,
        label,
        start,
        end
    ) in enumerate(
        periods,
        start=2
    ):

        print(
            f"\n[{i}/5] "
            f"{scenario} "
            f"{label} "
            f"{start[:4]}-{end[:4]}"
        )

        (
            datasets,
            pr,
            tmin,
            tmax,
            srad
        ) = open_model(
            model,
            scenario
        )

        try:

            wb3_future = compute_wb3_jjas(
                pr,
                tmin,
                tmax,
                srad,
                start,
                end
            )

        finally:

            close_all(
                datasets
            )

        spei_future = annual_min_spei(
            wb3_future,
            clim_mean,
            clim_std
        )

        scenario_dir = (
            SSP245_DIR
            if scenario == "ssp245"
            else SSP585_DIR
        )

        outfile = os.path.join(
            scenario_dir,
            f"{model}_{scenario}_{label}_"
            f"spei3_makkink_{start[:4]}-{end[:4]}.nc"
        )

        save_spei(
            spei_future,
            outfile
        )


def main():

    models = []

    suffix = (
        "_ssp245_prAdjust_CDFt-L-1V-0L.nc"
    )

    for filename in sorted(
        os.listdir(
            DIR_PR
        )
    ):

        if filename.endswith(
            suffix
        ):

            models.append(
                filename[
                    :-len(
                        suffix
                    )
                ]
            )

    print(
        "Modèles détectés:",
        len(
            models
        )
    )

    failures = []

    for model in models:

        try:

            process_model(
                model
            )

        except Exception as exc:

            print(
                "\nERREUR POUR",
                model,
                ":",
                repr(
                    exc
                )
            )

            failures.append(
                (
                    model,
                    str(
                        exc
                    )
                )
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FIN"
    )

    print(
        "=" * 80
    )

    if failures:

        print(
            "Échecs:"
        )

        for model, err in failures:

            print(
                f"  {model}: {err}"
            )

    else:

        print(
            "Tous les modèles ont été traités."
        )


if __name__ == "__main__":
    main()

