# =========================================
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
