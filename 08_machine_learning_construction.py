import os
import xarray as xr
import pandas as pd
import glob
import unicodedata

# =========================
# CLEAN
# =========================
def clean_dep(x):
    if pd.isna(x):
        return x
    x = str(x).lower().strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("utf-8")
    x = x.replace("-", " ").replace("_", " ")
    return x

# =========================
# PATHS
# =========================
BASE = "/media/sarr/01DC5DE9D15E8CF0/climate_extremes"
YIELD = "/media/sarr/01DC5DE9D15E8CF0/dossier/yield_data"
OUT   = "/media/sarr/01DC5DE9D15E8CF0/ml_datasets"

os.makedirs(OUT, exist_ok=True)

indices = ["TXx","TX35","RX1day","CDD","R95pTOT","PRtotal","SPEI3"]
crops = ["maize","millet","sorghum","groundnut"]
periods = ["historical","near_future","far_future"]

# =========================
# GET FILES
# =========================
def get_files(path, idx):
    return glob.glob(f"{path}/**/*{idx}.nc", recursive=True)

# =========================
# NETCDF → DF (ROBUSTE)
# =========================
def nc_to_df(file, idx):

    try:
        ds = xr.open_dataset(file, decode_times=False)

        # variable principale
        var = [v for v in ds.data_vars if "bnds" not in v.lower()][0]
        da = ds[var]

        # ===== CAS SPEI (year direct)
        if "year" in da.dims:
            years = ds["year"].values
            values = da.values.reshape(len(years), -1).mean(axis=1)

        # ===== CAS NORMAL (time)
        elif "time" in da.dims:
            values = da.values.reshape(len(ds["time"]), -1).mean(axis=1)

            try:
                years = xr.decode_cf(ds).time.dt.year.values
            except:
                years = pd.to_datetime(ds["time"].values).year

        else:
            return None

        if len(values) != len(years):
            return None

        df = pd.DataFrame({
            "year": years,
            idx: values
        })

        # =========================
        # EXTRACTION NOM FICHIER
        # =========================
        name = os.path.basename(file).replace(".nc","").split("_")

        # model (robuste)
        df["model"] = "unknown"
        for n in name:
            if any(m in n for m in ["ACCESS","BCC","CNRM","FGOALS","GFDL","MIROC","MPI","MRI","CCCma"]):
                df["model"] = n
                break

        # scénario
        scen = "historical"
        for n in name:
            if "ssp" in n:
                scen = n
        df["scenario"] = scen

        # département (robuste)
        df["department"] = "unknown"
        if "4km" in name:
            try:
                dep_index = name.index("4km") - 1
                df["department"] = clean_dep(name[dep_index])
            except:
                pass

        return df

    except Exception as e:
        print("❌", os.path.basename(file), "|", e)
        return None

# =========================
# LOOP
# =========================
for crop in crops:

    print("\n===", crop, "===")

    crop_out = f"{OUT}/{crop}"
    os.makedirs(crop_out, exist_ok=True)

    for period in periods:

        print("➡️", period)

        base_df = None

        for idx in indices:

            files = get_files(f"{BASE}/{crop}/cmip6/{period}", idx)
            print(idx, len(files))

            dfs = []

            for f in files:
                df = nc_to_df(f, idx)
                if df is not None:
                    dfs.append(df)

            if not dfs:
                print("❌ NO DATA:", idx)
                continue

            df_idx = pd.concat(dfs, ignore_index=True)

            df_idx = df_idx.groupby(
                ["year","department","model","scenario"],
                as_index=False
            )[idx].mean()

            if base_df is None:
                base_df = df_idx
            else:
                base_df = base_df.merge(
                    df_idx,
                    on=["year","department","model","scenario"],
                    how="outer"
                )

        if base_df is None:
            continue

        # =========================
        # FILTRE PÉRIODES
        # =========================
        if period == "historical":
            base_df = base_df[(base_df["year"] >= 2000) & (base_df["year"] <= 2013)]
        elif period == "near_future":
            base_df = base_df[(base_df["year"] >= 2036) & (base_df["year"] <= 2065)]
        elif period == "far_future":
            base_df = base_df[(base_df["year"] >= 2071) & (base_df["year"] <= 2100)]

        if base_df.empty:
            print("❌ EMPTY:", crop, period)
            continue

        # =========================
        # YIELD
        # =========================
        if period == "historical":

            yfile = f"{YIELD}/{crop}_yield.txt"

            if os.path.exists(yfile):

                y = pd.read_csv(yfile, sep="\t")

                y["department"] = y["department"].apply(clean_dep)
                base_df["department"] = base_df["department"].apply(clean_dep)

                base_df = base_df.merge(
                    y,
                    on=["year","department"],
                    how="left"
                )

                print("Yield merged OK")

            else:
                print("❌ YIELD FILE NOT FOUND:", yfile)

        # =========================
        # SAVE
        # =========================
        base_df = base_df.sort_values(
            ["model","year","department"]
        )

        out_file = f"{crop_out}/{period}.csv"
        base_df.to_csv(out_file, index=False)

        print("✅ SAVED:", out_file)

print("\nDONE")
