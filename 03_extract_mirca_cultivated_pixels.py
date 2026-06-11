import os
import glob

MIRCA_DIR = (
    "/media/sarr/01DC5DE9D15E8CF0/"
    "departments/agro_data/mask/monthly_growing_area_grids"
)

hdr_text = """NCOLS 720
NROWS 360
XLLCORNER -180
YLLCORNER -90
CELLSIZE 0.5
NODATA_VALUE -9999
NBITS 32
PIXELTYPE FLOAT
BYTEORDER LSBFIRST
"""

files = glob.glob(
    os.path.join(MIRCA_DIR, "*.flt")
)

for flt in files:

    hdr = flt.replace(".flt", ".hdr")

    with open(hdr, "w") as f:

        f.write(hdr_text)

    print("Created:", hdr)

print("\nDONE")# =========================================
# MIRCA MASK - CULTIVATED PIXELS EXTRACTION
# =========================================

import geopandas as gpd
import rioxarray as rxr
import pandas as pd
import numpy as np
import os
import unicodedata

# ==============================
# PATHS
# ==============================
shapefile_path = "/media/sarr/01DC5DE9D15E8CF0/departments/agro_data/SEN_adm/SEN_adm2.shp"

mirca_files = {
    "maize": "/media/sarr/01DC5DE9D15E8CF0/MIRCA/crop_02_rainfed.nc",
    "sorghum": "/media/sarr/01DC5DE9D15E8CF0/MIRCA/crop_05_rainfed.nc",
    "millet": "/media/sarr/01DC5DE9D15E8CF0/MIRCA/crop_06_rainfed.nc",
    "groundnut": "/media/sarr/01DC5DE9D15E8CF0/MIRCA/crop_12_rainfed.nc",
}

output_dir = "./mirca_pixel_outputs"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# CLEAN FUNCTION (IMPORTANT)
# ==============================
def clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("utf-8")
    return x.replace(" ", "_")

# ==============================
# READ SHAPEFILE
# ==============================
gdf_orig = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
gdf_orig["dep_clean"] = gdf_orig["NAME_2"].apply(clean_text)

print("Départements trouvés :", len(gdf_orig))

# ==============================
# LOOP CROPS
# ==============================
for crop_name, mirca_path in mirca_files.items():

    print(f"\nTraitement {crop_name}")

    mirca = rxr.open_rasterio(mirca_path, masked=True).squeeze()

    # reprojection propre (copie)
    gdf = gdf_orig.to_crs(mirca.rio.crs)

    all_pixels = []

    for _, row in gdf.iterrows():

        dept_name = row["dep_clean"]
        geom = [row.geometry]

        try:
            clipped = mirca.rio.clip(geom, gdf.crs, drop=True)
        except:
            continue

        cultivated = clipped.where(clipped > 0)

        if cultivated.count() == 0:
            continue

        df = cultivated.to_dataframe(name="crop_area").reset_index()
        df = df.dropna()

        df["department"] = dept_name
        df["crop"] = crop_name

        all_pixels.append(df)

    if len(all_pixels) == 0:
        print("Aucun pixel trouvé.")
        continue

    final_df = pd.concat(all_pixels, ignore_index=True)

    final_df = final_df[["department", "crop", "x", "y", "crop_area"]]
    final_df.rename(columns={"x": "lon", "y": "lat"}, inplace=True)

    output_file = os.path.join(output_dir, f"{crop_name}_cultivable_pixels.csv")
    final_df.to_csv(output_file, index=False)

    print(f"✔ Sauvegardé : {output_file}")

print("\nExtraction terminée pour les 4 cultures.")
