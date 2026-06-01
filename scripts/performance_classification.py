"""
CHAPITRE 4 — MESURES DE PERFORMANCE D'UN CLASSIFIEUR
====================================================
À partir de la MATRICE DE CONFUSION (cas binaire) :
                       prédit positif   prédit négatif
   réel positif            TP                FN
   réel négatif            FP                TN

FORMULES (cours) :
  Exactitude  = (TP+TN) / (TP+TN+FP+FN)     -> taux de bonnes classifications
  Précision   = TP / (TP+FP)                -> parmi les "positifs prédits", combien justes
  Rappel      = TP / (TP+FN)                -> parmi les vrais positifs, combien retrouvés
  Spécificité = TN / (TN+FP)                -> taux de vrais négatifs retrouvés
  F1-score    = 2*P*R / (P+R)               -> moyenne harmonique précision/rappel
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "rapport", "figures")
os.makedirs(FIG, exist_ok=True)


def mesures(tp, fn, fp, tn):
    exactitude = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    rappel = tp / (tp + fn) if (tp + fn) else 0.0
    specificite = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) else 0.0
    return exactitude, precision, rappel, specificite, f1


def main():
    print("=" * 70)
    print("MESURES DE PERFORMANCE — exemple")
    print("=" * 70)
    # exemple : 100 individus, classe "positif" = malade
    TP, FN, FP, TN = 40, 10, 5, 45
    print(f"Matrice de confusion : TP={TP}, FN={FN}, FP={FP}, TN={TN}")

    ex, pr, ra, sp, f1 = mesures(TP, FN, FP, TN)
    print(f"\nExactitude  = (TP+TN)/total = ({TP}+{TN})/{TP+FN+FP+TN} = {ex:.3f}")
    print(f"Précision   = TP/(TP+FP)    = {TP}/({TP}+{FP}) = {pr:.3f}")
    print(f"Rappel      = TP/(TP+FN)    = {TP}/({TP}+{FN}) = {ra:.3f}")
    print(f"Spécificité = TN/(TN+FP)    = {TN}/({TN}+{FP}) = {sp:.3f}")
    print(f"F1-score    = 2PR/(P+R)     = {f1:.3f}")

    # figure matrice de confusion
    import numpy as np

    M = np.array([[TP, FN], [FP, TN]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(M, cmap="Blues")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > M.max() / 2 else "black")
    ax.set_xticks([0, 1], ["prédit +", "prédit -"])
    ax.set_yticks([0, 1], ["réel +", "réel -"])
    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "matrice_confusion.png"), dpi=130)
    print("[figure] rapport/figures/matrice_confusion.png enregistrée")


if __name__ == "__main__":
    main()
