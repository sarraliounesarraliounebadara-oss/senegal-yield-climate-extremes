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

print("\n✅ MIRCA NetCDF conversion completed")# =========================================
# BUILD CMIP6 CROP MASKS FROM MIRCA PIXELS
# =========================================

import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import unicodedata

# =====================================================
# PATHS
# =====================================================

MASK_TXT_DIR = "/media/sarr/01DC5DE7BBEA1120/Donnees/new_scripts/mirca_outputs_4km"
CMIP6 = "/media/sarr/01DC5DE9D15E8CF0/departments/climate_data/cmip6"
OUT = "/media/sarr/01DC5DE7BBEA1120/Donnees/python-script/crop_masks_nc"

os.makedirs(OUT, exist_ok=True)

# =====================================================
# CLEAN FUNCTION
# =====================================================

def clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("utf-8")
    return x.replace(" ", "_")

# =====================================================
# READ CMIP6 GRID
# =====================================================

example_file = glob.glob(os.path.join(CMIP6, "pr", "*.nc"))[0]
ds = xr.open_dataset(example_file)

lat = ds.lat.values
lon = ds.lon.values

ds.close()

print("Grid size:", len(lat), "x", len(lon))

# =====================================================
# CROPS
# =====================================================

crops = ["maize", "millet", "sorghum", "groundnut"]

# =====================================================
# LOOP
# =====================================================

for crop in crops:

    print("\n==============================")
    print("Crop:", crop)

    files = glob.glob(os.path.join(MASK_TXT_DIR, f"{crop}_*_pixels.txt"))

    if not files:
        print("⚠ Aucun fichier trouvé")
        continue

    for f in files:

        dept = os.path.basename(f)
        dept = dept.replace(f"{crop}_", "").replace("_pixels.txt", "")
        dept = clean_text(dept)

        print("Department:", dept)

        df = pd.read_csv(f, sep=r"\s+")

        if df.empty or not {"lat", "lon"}.issubset(df.columns):
            continue

        # =====================================================
        # GRID INDEX (vectorized, safe)
        # =====================================================

        lat_idx = np.searchsorted(lat, df["lat"].values)
        lon_idx = np.searchsorted(lon, df["lon"].values)

        lat_idx = np.clip(lat_idx, 0, len(lat)-1)
        lon_idx = np.clip(lon_idx, 0, len(lon)-1)

        # =====================================================
        # MASK (weighted possible)
        # =====================================================

        mask = np.zeros((len(lat), len(lon)), dtype=np.float32)

        for i in range(len(df)):
            mask[lat_idx[i], lon_idx[i]] += 1

        # binaire (optionnel)
        mask = (mask > 0).astype(np.int8)

        # =====================================================
        # DATAARRAY
        # =====================================================

        da = xr.DataArray(
            mask,
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
            name="mask"
        )

        da.attrs["description"] = "Crop mask from MIRCA2000"
        da.attrs["values"] = "1=crop area, 0=outside"

        ds_out = da.to_dataset()

        out_file = os.path.join(OUT, f"mask_{crop}_{dept}.nc")
        ds_out.to_netcdf(out_file)

        print("✔ Saved:", out_file)

print("\n✅ Masks completed")
