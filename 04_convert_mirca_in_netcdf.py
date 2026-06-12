# =========================================
# CONVERT MIRCA FLT -> NETCDF
# FINAL MEMORY-SAFE VERSION
# =========================================

import xarray as xr
import numpy as np
import os
import gc

# =========================================
# PATHS
# =========================================

MIRCA_DIR = (
    "/media/sarr/01DC5DE9D15E8CF0/"
    "departments/agro_data/mask/monthly_growing_area_grids"
)

OUTDIR = (
    "/media/sarr/01DC5DE9D15E8CF0/MIRCA"
)

os.makedirs(OUTDIR, exist_ok=True)

# =========================================
# MIRCA FILES
# =========================================

mirca_files = {

    "crop_02_rainfed_12.flt":
    "crop_02_rainfed.nc",

    "crop_05_rainfed_12.flt":
    "crop_05_rainfed.nc",

    "crop_06_rainfed_12.flt":
    "crop_06_rainfed.nc",

    "crop_12_rainfed_12.flt":
    "crop_12_rainfed.nc",
}

# =========================================
# GRID PARAMETERS
# =========================================

ncols = 4320
nrows = 2160
nmonths = 12

cellsize = 5 / 60  # 0.083333333°

nodata = -9999

# =========================================
# COORDINATES
# =========================================

lon = np.arange(
    -180 + cellsize / 2,
    180,
    cellsize,
    dtype=np.float32
)

lat = np.arange(
    90 - cellsize / 2,
    -90,
    -cellsize,
    dtype=np.float32
)

month = np.arange(
    1,
    13,
    dtype=np.int16
)

print(
    "Grid size:",
    len(lat),
    "x",
    len(lon)
)

# =========================================
# LOOP FILES
# =========================================

for flt_name, nc_name in mirca_files.items():

    print("\n==============================")
    print("Processing:", flt_name)

    flt_path = os.path.join(
        MIRCA_DIR,
        flt_name
    )

    if not os.path.exists(flt_path):

        print("Missing:", flt_path)

        continue

    # =========================================
    # OPEN MEMMAP READ-ONLY
    # =========================================

    print("Opening memmap...")

    data = np.memmap(

        flt_path,

        dtype=np.float32,

        mode="r",

        shape=(
            nmonths,
            nrows,
            ncols
        )

    )

    # =========================================
    # BUILD XARRAY DIRECTLY
    # =========================================

    da = xr.DataArray(

        data,

        coords={

            "month": month,
            "lat": lat,
            "lon": lon

        },

        dims=(

            "month",
            "lat",
            "lon"

        ),

        name="crop_area"

    )

    # =========================================
    # CLEAN WITHOUT COPYING
    # =========================================

    print("Cleaning values...")

    da = da.where(
        da != nodata
    )

    da = da.where(
        da >= 0
    )

    print(
        "Min:",
        float(da.min(skipna=True))
    )

    print(
        "Max:",
        float(da.max(skipna=True))
    )

    # =========================================
    # ATTRIBUTES
    # =========================================

    da.attrs["description"] = (
        "MIRCA2000 monthly rainfed growing area"
    )

    da.attrs["units"] = "ha"

    # =========================================
    # DATASET
    # =========================================

    ds = da.to_dataset()

    # =========================================
    # NETCDF COMPRESSION
    # =========================================

    encoding = {

        "crop_area": {

            "dtype": "float32",
            "zlib": True,
            "complevel": 4

        }

    }

    # =========================================
    # SAVE
    # =========================================

    outpath = os.path.join(
        OUTDIR,
        nc_name
    )

    print("Saving NetCDF...")

    ds.to_netcdf(

        outpath,

        engine="netcdf4",

        encoding=encoding

    )

    print("✔ Saved:", outpath)

    # =========================================
    # FREE MEMORY
    # =========================================

    del data
    del da
    del ds

    gc.collect()

print("\n✅ MIRCA NetCDF conversion completed")
