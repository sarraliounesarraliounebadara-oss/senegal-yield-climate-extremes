#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPEI3-Makkink OBSERVATIONS - VERSION CORRIGÉE

Correction importante :
la standardisation mensuelle est appliquée mois par mois sans créer
une dimension supplémentaire "month".

Sortie attendue :
  SPEI3(time, Lat, Lon)
avec une seule valeur par pixel et par année.
"""

import os
import numpy as np
import xarray as xr

BASE_CANDIDATES = [
    "/mnt/disque/departments/climate_data/anacim_data",
    "/media/sarr/01DC5DE9D15E8CF0/departments/climate_data/anacim_data",
]

BASE = None
for candidate in BASE_CANDIDATES:
    if os.path.isdir(candidate):
        BASE = candidate
        break

if BASE is None:
    BASE = "/mnt/disque/departments/climate_data/anacim_data"

PR_FILE = os.path.join(BASE, "ARC2_merged.nc")
TMIN_FILE = os.path.join(BASE, "CHIRTS_Tmin_merged.nc")
TMAX_FILE = os.path.join(BASE, "CHIRTS_Tmax_merged.nc")
SRAD_FILE = os.path.join(BASE, "srad_HargreavesSamani_obs_1984-2013.nc")

OUT_FILE = os.path.join(
    BASE,
    "observations_spei3_makkink_1984-2013.nc"
)

HIST_START = "1986-01-01"
HIST_END   = "2013-12-31"

JJAS = [6, 7, 8, 9]

MAKKINK_A = 0.61
MAKKINK_B = -0.12
PRESSURE_KPA = 101.3
GAMMA_PSY = 0.000665 * PRESSURE_KPA

CHUNKS = {
    "time": 366,
    "Lat": 41,
    "Lon": 57,
}


def compute_pet_makkink(tmin_month, tmax_month, srad_month):

    tmean = (tmin_month + tmax_month) / 2.0

    es = 0.6108 * np.exp(
        (17.27 * tmean)
        / (tmean + 237.3)
    )

    delta = (
        4098.0 * es
        / ((tmean + 237.3) ** 2)
    )

    lambda_v = (
        2.501
        - 0.002361 * tmean
    )

    pet_daily = (
        MAKKINK_A
        * (delta / (delta + GAMMA_PSY))
        * (srad_month / lambda_v)
        + MAKKINK_B
    )

    pet_daily = pet_daily.clip(min=0.0)

    days = xr.DataArray(
        tmin_month.time.dt.days_in_month,
        dims=["time"],
        coords={"time": tmin_month.time}
    )

    pet_month = pet_daily * days
    pet_month.name = "PET_Makkink"

    return pet_month


def standardize_monthly(wb3):
    """
    Standardisation correcte :
      1) soustraction de la moyenne du mois correspondant
      2) division par l'écart-type du mois correspondant

    IMPORTANT :
    il faut refaire groupby("time.month") avant la division.
    Sinon xarray ajoute une dimension indépendante "month".
    """

    clim_mean = wb3.groupby(
        "time.month"
    ).mean("time")

    clim_std = wb3.groupby(
        "time.month"
    ).std("time")

    clim_std = xr.where(
        clim_std < 1.0e-6,
        1.0e-6,
        clim_std
    )

    anomaly = (
        wb3.groupby("time.month")
        - clim_mean
    )

    spei = (
        anomaly.groupby("time.month")
        / clim_std
    )

    return spei


def main():

    print("=" * 70)
    print("SPEI3-MAKKINK OBSERVATIONS - CORRIGÉ")
    print("=" * 70)

    for path in [
        PR_FILE,
        TMIN_FILE,
        TMAX_FILE,
        SRAD_FILE
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    ds_pr = xr.open_dataset(
        PR_FILE,
        chunks=CHUNKS,
        decode_times=True
    )

    ds_tmin = xr.open_dataset(
        TMIN_FILE,
        chunks=CHUNKS,
        decode_times=True
    )

    ds_tmax = xr.open_dataset(
        TMAX_FILE,
        chunks=CHUNKS,
        decode_times=True
    )

    ds_srad = xr.open_dataset(
        SRAD_FILE,
        chunks=CHUNKS,
        decode_times=True
    )

    try:

        pr = ds_pr["precip"].sel(
            time=slice(HIST_START, HIST_END)
        )

        tmin = ds_tmin["temp"].sel(
            time=slice(HIST_START, HIST_END)
        )

        tmax = ds_tmax["temp"].sel(
            time=slice(HIST_START, HIST_END)
        )

        srad = ds_srad["srad"].sel(
            time=slice(HIST_START, HIST_END)
        )

        pr, tmin, tmax, srad = xr.align(
            pr,
            tmin,
            tmax,
            srad,
            join="inner"
        )

        print(
            "Dates communes:",
            str(pr.time.values[0]),
            "->",
            str(pr.time.values[-1])
        )

        print(
            "Nombre de jours:",
            pr.sizes["time"]
        )

        # ---------------------------------------------
        # Monthly aggregation
        # ---------------------------------------------

        print("Agrégation mensuelle...")

        pr_month = pr.resample(
            time="MS"
        ).sum(skipna=False)

        tmin_month = tmin.resample(
            time="MS"
        ).mean(skipna=False)

        tmax_month = tmax.resample(
            time="MS"
        ).mean(skipna=False)

        srad_month = srad.resample(
            time="MS"
        ).mean(skipna=False)

        (
            pr_month,
            tmin_month,
            tmax_month,
            srad_month
        ) = xr.align(
            pr_month,
            tmin_month,
            tmax_month,
            srad_month,
            join="inner"
        )

        # ---------------------------------------------
        # PET
        # ---------------------------------------------

        print("Calcul PET-Makkink...")

        pet_month = compute_pet_makkink(
            tmin_month,
            tmax_month,
            srad_month
        )

        # ---------------------------------------------
        # Water balance and SPEI3
        # ---------------------------------------------

        wb = pr_month - pet_month

        wb3 = wb.rolling(
            time=3,
            min_periods=3
        ).sum()

        print("Standardisation mensuelle...")

        spei3_monthly = standardize_monthly(
            wb3
        )

        # Vérification essentielle
        if "month" in spei3_monthly.dims:

            raise RuntimeError(
                "ERREUR: une dimension 'month' supplémentaire "
                "a été créée dans SPEI3."
            )

        # ---------------------------------------------
        # JJAS
        # ---------------------------------------------

        spei3_jjas = spei3_monthly.where(
            spei3_monthly.time.dt.month.isin(JJAS),
            drop=True
        )

        # ---------------------------------------------
        # Minimum JJAS
        # ---------------------------------------------

        spei3 = spei3_jjas.groupby(
            "time.year"
        ).min(
            "time",
            skipna=False
        )

        if "month" in spei3.dims:

            raise RuntimeError(
                "ERREUR: 'month' ne doit pas être une dimension "
                "du SPEI3 annuel."
            )

        # ---------------------------------------------
        # Year -> time
        # ---------------------------------------------

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

        spei3.attrs = {
            "long_name": "Minimum JJAS SPEI3 based on Makkink PET",
            "units": "1",
            "PET_method": "Makkink",
            "SPEI_scale": "3 months",
            "season": "JJAS",
            "seasonal_aggregation": "minimum monthly SPEI3 during JJAS",
            "reference_period": "1984-2013",
            "standardization": (
                "monthly historical mean and standard deviation "
                "of 3-month climatic water balance"
            ),
            "radiation_source": (
                "Hargreaves-Samani estimated radiation from CHIRTS Tmin/Tmax"
            ),
            "radiation_kRs": 0.16,
        }

        print("Dimensions finales avant calcul:")
        print(spei3.dims)
        print(spei3.sizes)

        spei3 = spei3.astype(
            np.float32
        ).compute()

        print(
            "Min / Mean / Max:",
            float(spei3.min(skipna=True)),
            float(spei3.mean(skipna=True)),
            float(spei3.max(skipna=True))
        )

        print(
            "Nombre d'années:",
            spei3.sizes["time"]
        )

        spei3.to_netcdf(
            OUT_FILE,
            encoding={
                "SPEI3": {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32"
                }
            }
        )

        print("Saved:", OUT_FILE)

    finally:

        ds_pr.close()
        ds_tmin.close()
        ds_tmax.close()
        ds_srad.close()


if __name__ == "__main__":
    main()

