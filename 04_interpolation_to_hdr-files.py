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

print("\nDONE")
