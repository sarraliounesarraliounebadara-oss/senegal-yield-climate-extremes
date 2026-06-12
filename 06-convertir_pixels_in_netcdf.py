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

print("\n✅ ALL NETCDF MASKS CREATED")
