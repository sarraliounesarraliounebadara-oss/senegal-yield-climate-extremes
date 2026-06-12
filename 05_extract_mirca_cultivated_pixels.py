# =========================================
# BUILD MIRCA CROP MASKS
# WITHOUT INTERPOLATION
# =========================================

import xarray as xr
import geopandas as gpd
import rioxarray
import numpy as np
import pandas as pd
import os
import unicodedata

# =========================================
# PATHS
# =========================================

SHAPEFILE = (
    "/media/sarr/01DC5DE9D15E8CF0/"
    "departments/agro_data/SEN_adm/SEN_adm2.shp"
)

MIRCA_DIR = (
    "/media/sarr/01DC5DE9D15E8CF0/MIRCA"
)

OUTDIR = (
    "/media/sarr/01DC5DE7BBEA1120/"
    "Donnees/mirca_masks"
)

os.makedirs(OUTDIR, exist_ok=True)

# =========================================
# MIRCA FILES
# =========================================

mirca_files = {

    "maize":
    "crop_02_rainfed.nc",

    "sorghum":
    "crop_05_rainfed.nc",

    "millet":
    "crop_06_rainfed.nc",

    "groundnut":
    "crop_12_rainfed.nc"
}

# =========================================
# STUDY DEPARTMENTS
# =========================================

study_departments = [

    "bakel",
    "bambey",
    "bignona",
    "diourbel",
    "fatick",
    "foundiougne",
    "gossas",
    "kaffrine",
    "kaolack",
    "kebemer",
    "kedougou",
    "kolda",
    "linguere",
    "louga",
    "mbacke",
    "mbour",
    "nioro_du_rip",
    "oussouye",
    "sedhiou",
    "tambacounda",
    "thies",
    "tivaouane",
    "velingara",
    "ziguinchor"
]

# =========================================
# MINIMUM CULTIVATED AREA
# =========================================

MIN_AREA = 0

# =========================================
# CLEAN FUNCTION
# =========================================

def clean_text(x):

    if pd.isna(x):
        return x

    x = str(x).lower().strip()

    x = unicodedata.normalize(
        "NFKD",
        x
    ).encode(
        "ascii",
        "ignore"
    ).decode("utf-8")

    return x.replace(" ", "_")

# =========================================
# READ SHAPEFILE
# =========================================

gdf = gpd.read_file(
    SHAPEFILE
)

gdf = gdf.to_crs(
    "EPSG:4326"
)

gdf["department"] = (

    gdf["NAME_2"]
    .apply(clean_text)
    .str.strip("_")

)

# =========================================
# KEEP ONLY STUDY AREA
# =========================================

gdf = gdf[
    gdf["department"].isin(
        study_departments
    )
]

print(
    "Departments:",
    len(gdf)
)

# =========================================
# LOOP CROPS
# =========================================

for crop, filename in mirca_files.items():

    print("\n==============================")
    print("Crop:", crop)

    path = os.path.join(
        MIRCA_DIR,
        filename
    )

    print("Opening:", path)

    # =====================================
    # OPEN DATASET
    # =====================================

    ds = xr.open_dataset(path)

    da = ds["crop_area"]

    # =====================================
    # MEAN OVER MONTHS
    # =====================================

    print("Computing mean growing area...")

    da = da.mean(
        dim="month",
        skipna=True
    )

    # =====================================
    # RENAME FOR RIOXARRAY
    # =====================================

    da = da.rename({

        "lon": "x",
        "lat": "y"

    })

    # =====================================
    # DEFINE SPATIAL DIMS
    # =====================================

    da = da.rio.set_spatial_dims(

        x_dim="x",
        y_dim="y"

    )

    da = da.rio.write_crs(
        "EPSG:4326"
    )

    # =====================================
    # REMOVE NEGATIVE VALUES
    # =====================================

    da = da.where(
        da > 0
    )

    # =====================================
    # LOOP DEPARTMENTS
    # =====================================

    for _, row in gdf.iterrows():

        dept = row["department"]

        print(
            "Department:",
            dept
        )

        geom = [row.geometry]

        # =================================
        # CLIP
        # =================================

        try:

            clipped = da.rio.clip(

                geom,
                gdf.crs,
                drop=True

            )

        except Exception as e:

            print(
                "Clip error:",
                str(e)
            )

            continue

        # =================================
        # TO DATAFRAME
        # =================================

        df = clipped.to_dataframe(
            name="crop_area"
        ).reset_index()

        # =================================
        # RENAME COORDS
        # =================================

        df = df.rename(columns={

            "x": "lon",
            "y": "lat"

        })

        # =================================
        # REMOVE NAN
        # =================================

        df = df.dropna()

        # =================================
        # KEEP CULTIVATED PIXELS
        # =================================

        df = df[
            df["crop_area"] >= MIN_AREA
        ]

        if df.empty:

            print(
                "No cultivated pixels"
            )

            continue

        # =================================
        # COMPUTE WEIGHTS
        # =================================

        total = df["crop_area"].sum()

        df["weight"] = (

            df["crop_area"]
            / total

        )

        # =================================
        # ADD METADATA
        # =================================

        df["crop"] = crop

        df["department"] = dept

        # =================================
        # KEEP COLUMNS
        # =================================

        df = df[

            [

                "department",
                "crop",
                "lon",
                "lat",
                "crop_area",
                "weight"

            ]

        ]

        # =================================
        # SORT
        # =================================

        df = df.sort_values(
            ["lat", "lon"]
        )

        # =================================
        # SAVE
        # =================================

        outfile = os.path.join(

            OUTDIR,

            f"{crop}_{dept}_mask.csv"

        )

        df.to_csv(

            outfile,

            index=False

        )

        print(
            "✔ Saved:",
            outfile,
            "| pixels:",
            len(df)
        )

    ds.close()

print("\n✅ ALL MASKS COMPLETED")
