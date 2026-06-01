"""
CHAPITRE 2 — PRÉTRAITEMENT DES DONNÉES
=======================================
Ce script illustre, pas à pas, deux transformations du cours :
  1. La NORMALISATION min-max         -> ramène les valeurs dans [0, 1]
  2. La STANDARDISATION (z-score)     -> centre (moyenne 0) et réduit (écart-type 1)
Et il résout l'EXERCICE 1 (détection des problèmes de qualité d'un tableau).

On recode les formules "à la main" pour bien comprendre, puis on vérifie.
Les figures sont enregistrées dans rapport/figures/ pour le rapport LaTeX.
"""

import os

import matplotlib

matplotlib.use("Agg")  # backend sans écran : on enregistre les figures en PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = os.path.join(os.path.dirname(__file__), "..", "rapport", "figures")
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. NORMALISATION MIN-MAX
# ---------------------------------------------------------------------------
# Formule du cours :  X* = (X - min(X)) / (max(X) - min(X))
# Après transformation, toutes les valeurs sont dans l'intervalle [0, 1].
def normalisation_min_max(x: np.ndarray) -> np.ndarray:
    xmin, xmax = x.min(), x.max()
    return (x - xmin) / (xmax - xmin)


# ---------------------------------------------------------------------------
# 2. STANDARDISATION (Z-SCORE)
# ---------------------------------------------------------------------------
# Formule du cours :  X* = (X - moyenne(X)) / ecart_type(X)
#   moyenne   : x_barre = (1/N) * somme(x_i)
#   variance  : sigma^2 = (1/N) * somme((x_i - x_barre)^2)   (écart-type = racine)
# Résultat : moyenne ~ 0 et écart-type ~ 1 (valeurs souvent entre -3 et +3).
def standardisation_zscore(x: np.ndarray) -> np.ndarray:
    moyenne = x.mean()
    ecart_type = x.std()  # std de numpy = écart-type "population" (divise par N)
    return (x - moyenne) / ecart_type


def demo_transformations() -> None:
    print("=" * 70)
    print("1) NORMALISATION MIN-MAX et 2) STANDARDISATION Z-SCORE")
    print("=" * 70)

    # Petit jeu de valeurs (ex. "day minutes" simplifié) pour voir les calculs
    x = np.array([200.0, 150.0, 300.0, 100.0, 250.0])
    print("Valeurs originales        :", x)

    # --- min-max détaillé ---
    xmin, xmax = x.min(), x.max()
    print(f"\nmin(X) = {xmin}, max(X) = {xmax}, étendue = {xmax - xmin}")
    xn = normalisation_min_max(x)
    for xi, xni in zip(x, xn):
        print(f"  ({xi:6.1f} - {xmin}) / ({xmax} - {xmin}) = {xni:.3f}")
    print("Normalisé (min-max)       :", np.round(xn, 3),
          "-> bien dans [0, 1] :", xn.min() >= 0 and xn.max() <= 1)

    # --- z-score détaillé ---
    moyenne, sigma = x.mean(), x.std()
    print(f"\nmoyenne = {moyenne}, écart-type (sigma) = {sigma:.3f}")
    xs = standardisation_zscore(x)
    for xi, xsi in zip(x, xs):
        print(f"  ({xi:6.1f} - {moyenne}) / {sigma:.3f} = {xsi:+.3f}")
    print("Standardisé (z-score)     :", np.round(xs, 3))
    print(f"  -> moyenne ≈ {xs.mean():.3f}, écart-type ≈ {xs.std():.3f}")

    # --- figure comparative (comme demandé : "vérifier par une courbe") ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, data, titre, col in zip(
        axes,
        [x, xn, xs],
        ["Originales", "Min-max  ∈ [0,1]", "Z-score (centré-réduit)"],
        ["#4C72B0", "#55A868", "#C44E52"],
    ):
        ax.plot(data, "o-", color=col)
        ax.set_title(titre)
        ax.grid(alpha=0.3)
    axes[1].axhline(0, ls="--", c="grey")
    axes[1].axhline(1, ls="--", c="grey")
    fig.suptitle("Effet des transformations sur les mêmes données")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "chap2_transformations.png"), dpi=130)
    print("\n[figure] rapport/figures/chap2_transformations.png enregistrée")


# ---------------------------------------------------------------------------
# EXERCICE 1 — Détecter les problèmes de qualité d'un tableau
# ---------------------------------------------------------------------------
def exercice1_qualite() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 1 — Problèmes de qualité des données")
    print("=" * 70)
    df = pd.DataFrame(
        {
            "ID_client": [1001, 1002, 1003, 1004, 1005],
            "Code_postal": ["10048", "J2S7K7", "90210", "6269", "55101"],
            "Sexe": ["M", "F", None, "M", "F"],
            "Revenu": [75000, -40000, 10000000, 50000, 999999],
            "Age": ["C", "40", "45", "0", "30"],  # 'C' = valeur non numérique
            "Etat_civil": ["M", "V", "C", "C", "D"],
            "Transaction": [5000, 4000, 7000, 1000, 3000],
        }
    )
    print(df.to_string(index=False))

    print("\nProblèmes détectés automatiquement :")
    # Sexe manquant
    for i in df.index[df["Sexe"].isna()]:
        print(f"  - Ligne {df.loc[i,'ID_client']}: SEXE manquant (valeur nulle).")
    # Revenu négatif ou aberrant
    for i in df.index[df["Revenu"] < 0]:
        print(f"  - Ligne {df.loc[i,'ID_client']}: REVENU négatif ({df.loc[i,'Revenu']}) -> impossible.")
    for i in df.index[df["Revenu"] > 1_000_000]:
        print(f"  - Ligne {df.loc[i,'ID_client']}: REVENU aberrant ({df.loc[i,'Revenu']}) -> valeur extrême.")
    for i in df.index[df["Revenu"] == 999999]:
        print(f"  - Ligne {df.loc[i,'ID_client']}: REVENU = 999999 -> code 'valeur manquante' déguisé.")
    # Age non numérique ou impossible
    for i in df.index:
        age = df.loc[i, "Age"]
        if not str(age).isdigit():
            print(f"  - Ligne {df.loc[i,'ID_client']}: AGE='{age}' non numérique (incohérence de type).")
        elif int(age) == 0:
            print(f"  - Ligne {df.loc[i,'ID_client']}: AGE=0 -> probablement manquant/erroné.")
    # Code postal incohérent (format mixte chiffres / lettres, longueurs variables)
    for i in df.index:
        cp = df.loc[i, "Code_postal"]
        if not cp.isdigit():
            print(f"  - Ligne {df.loc[i,'ID_client']}: CODE_POSTAL='{cp}' format non numérique (US vs Canada).")
        elif len(cp) != 5:
            print(f"  - Ligne {df.loc[i,'ID_client']}: CODE_POSTAL='{cp}' longueur ≠ 5 (incohérent).")

    print(
        "\nRésumé : valeurs manquantes (sexe), valeurs hors domaine (revenu négatif),\n"
        "valeurs aberrantes/extrêmes (10 000 000), codes de manquant déguisés (999999, age 0),\n"
        "et incohérences de format/codage (codes postaux US/Canada mélangés)."
    )


if __name__ == "__main__":
    demo_transformations()
    exercice1_qualite()
