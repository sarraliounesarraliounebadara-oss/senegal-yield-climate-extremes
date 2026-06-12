#!/usr/bin/env python3
# =========================================================================
# BUILD YIELD ANOMALY (LOESS DETRENDING) - VERSION CORRIGÉE ET SÉCURISÉE
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
    # CHARGEMENT MULTI-FORMATS ULTRA-ROBUSTE (Excel / Tab / CSV)
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

    # Si après tous les essais aucun DataFrame n'est valide
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
    # CORRECTION CRITIQUE : NETTOYAGE DES VALEURS MANQUANTES / FAUX ZÉROS
    # -----------------------------------------------------------------
    # 1. Suppression préventive des lignes n'ayant ni année ni rendement (NaN)
    df = df.dropna(subset=["yield", "year"])
    
    # 2. Remplacement des rendements à 0.0 par des NaN (données manquantes réelles)
    # Cela évite les fausses anomalies à -100% et stabilise la tendance LOESS
    df = df[df["yield"] > 0]

    # Tri des données par département puis par ordre chronologique
    df = df.sort_values(["department", "year"]).reset_index(drop=True)

    # Initialisation des colonnes de résultats
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

        # Lissage LOESS (frac=0.4)
        loess_fit = lowess(
            group["yield"], group["year"], frac=0.4, return_sorted=True
        )

        # Interpolation pour s'assurer d'avoir une valeur de tendance pour chaque année du groupe
        trend_all = np.interp(group["year"], loess_fit[:, 0], loess_fit[:, 1])

        # -----------------------------------------------------------------
        # SÉCURITÉ MATRICIELLE ET SÉCURISATION CONTRE LA DIVISION PAR 0
        # -----------------------------------------------------------------
        y_val = group["yield"].values
        t_val = trend_all

        # SÉCURITÉ 1 : Forcer les tendances négatives de LOESS à être au moins égales à 0.0
        t_val = np.where(t_val < 0, np.nan, t_val)

        # Sauvegarde de la tendance corrigée dans le DataFrame principal
        df.loc[idx, "yield_trend"] = t_val

        # Initialisation du vecteur d'anomalies temporaire rempli de NaN
        anomaly = np.full(len(group), np.nan)

        # SÉCURITÉ 2 : Seuil limite sous lequel le calcul de % n'est plus viable
       # Initialisation du vecteur d'anomalies temporaire rempli de NaN
        anomaly = np.full(len(group), np.nan)

        # SÉCURITÉ 2 : Seuil limite sous lequel le calcul de % n'est plus viable
        seuil_tendance = 10.0

        # Masques logiques robustes aux NaN
        # 1. La tendance est valide (pas NaN) et supérieure au seuil
        tendance_saine = (~np.isnan(t_val)) & (t_val > seuil_tendance)
        
        # 2. La tendance est valide (pas NaN), et la tendance ET le rendement sont inférieurs au seuil
        tendance_et_rendement_nuls = (~np.isnan(t_val)) & (t_val <= seuil_tendance) & (y_val <= seuil_tendance)

        # Application des calculs vectorisés
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

    # Enregistrement au format texte séparé par des tabulations (TSV)
    df.to_csv(out_file, sep="\t", index=False)
    print(f"✅ Fichier anomalies enregistré avec succès : {out_file}")

print("\n🎯 TRAITEMENT GLOBAL DE TOUTES LES CULTURES TERMINÉ")
