#!/usr/bin/env python3
# =========================================================================
# BUILD YIELD ANOMALY (LOESS DETRENDING) - 
# =========================================================================

import os
import unicodedata
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

# ==========================
# CONFIGURATION DES CHEMINS
# ==========================
BASE = "/media/sarr/01DC5DE9D15E8CF0/dossier/yield_data"

files = [
    "groundnut_yield.xlsx",
    "maize_yield.xlsx",
    "millet_yield.xlsx",
    "sorghum_yield.xlsx",
]


# ==========================
# FONCTION DE NETTOYAGE
# ==========================
def clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).lower().strip()
    x = (
        unicodedata.normalize("NFKD", x)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )
    return x.replace(" ", "_")


# ==========================
# BOUCLE SUR LES CULTURES
# ==========================
for f in files:
    path = os.path.join(BASE, f)

    if not os.path.exists(path):
        print(f"⚠️ Fichier introuvable, passé : {path}")
        continue

    print("\n=================================")
    print("Processing:", f)
    print("=================================")

    # =================================================================
    # =================================================================
    df = None
    
    # 1. Tentative en format Excel réel
    try:
        df = pd.read_excel(path, engine="openpyxl")
        df.columns = [c.lower().strip() for c in df.columns]
        if "department" not in df.columns:
            df = None  # Forcer le passage au bloc de secours textuel
    except Exception:
        df = None

    # 2. Si ce n'est pas un vrai Excel, on teste les formats texte courants
    if df is None:
        for separateur in ["\t", ";", ","]:
            try:
                df_test = pd.read_csv(path, sep=separateur)
                df_test.columns = [c.lower().strip() for c in df_test.columns]
                
                # Si on trouve la colonne clé, c'est le bon séparateur !
                if "department" in df_test.columns:
                    df = df_test
                    print(f"ℹ️ Format détecté : Fichier texte (Séparateur : '{separateur.encode('unicode_escape').decode()}')")
                    break
            except Exception:
                continue

    if df is None:
        print(f"❌ Impossible de lire le fichier {f} ou de détecter la colonne 'department'. Fichier ignoré.")
        continue

    # ==========================
    # NETTOYAGE & VÉRIFICATION
    # ==========================
    # S'assurer que toutes les colonnes vitales sont présentes
    if (
        "department" not in df.columns
        or "yield" not in df.columns
        or "year" not in df.columns
    ):
        print(f"❌ Colonnes requises manquantes dans {f} (Colonnes lues : {list(df.columns)}). Fichier ignoré.")
        continue

    df["department"] = df["department"].apply(clean_text)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    df = df.dropna(subset=["yield", "year"])
    
    
    df = df[df["yield"] > 0]

    df = df.sort_values(["department", "year"]).reset_index(drop=True)

    df["yield_trend"] = np.nan
    df["yield_anomaly"] = np.nan

    # ==========================
    # LOESS PAR DÉPARTEMENT
    # ==========================
    for dep, group in df.groupby("department"):
        idx = group.index

        # Vérification du nombre minimal de points pour faire tourner LOESS
        if len(group) < 6:
            print(f"   Department: {dep} -> Pas assez de données ({len(group)} obs pour LOESS)")
            continue

        # Lissage LOESS (frac=0.6)
        loess_fit = lowess(
            group["yield"], group["year"], frac=0.6, return_sorted=True
        )

        # Interpolation pour s'assurer d'avoir une valeur de tendance pour chaque année du groupe
        trend_all = np.interp(group["year"], loess_fit[:, 0], loess_fit[:, 1])

        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        y_val = group["yield"].values
        t_val = trend_all

        t_val = np.where(t_val < 0, np.nan, t_val)

        df.loc[idx, "yield_trend"] = t_val

        anomaly = np.full(len(group), np.nan)

        anomaly = np.full(len(group), np.nan)

        seuil_tendance = 10.0

        tendance_saine = (~np.isnan(t_val)) & (t_val > seuil_tendance)
        
        tendance_et_rendement_nuls = (~np.isnan(t_val)) & (t_val <= seuil_tendance) & (y_val <= seuil_tendance)

        anomaly[tendance_saine] = (
            (y_val[tendance_saine] - t_val[tendance_saine]) * 100
        ) / t_val[tendance_saine]
        
        anomaly[tendance_et_rendement_nuls] = 0.0

        # Sauvegarde dans le DataFrame principal
        df.loc[idx, "yield_anomaly"] = anomaly

    # ==========================
    # SAUVEGARDE DES RÉSULTATS
    # ==========================
    base_path, _ = os.path.splitext(path)
    out_file = base_path + ".txt"

    df.to_csv(out_file, sep="\t", index=False)
    print(f"✅ Fichier anomalies enregistré avec succès : {out_file}")

print("\n🎯 TRAITEMENT GLOBAL DE TOUTES LES CULTURES TERMINÉ")
