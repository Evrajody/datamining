"""
CHAPITRE 5 — CLUSTERING HIÉRARCHIQUE ASCENDANT (lien simple)
============================================================
EXERCICE : à partir de la matrice de distance ci-dessous, réaliser le
clustering hiérarchique ascendant avec le LIEN SIMPLE (single linkage).

Matrice de distance (symétrique) :
        B   C   D   E
   A    1   3   2   4
   B        3   2   3
   C            1   3
   D                5

LIEN SIMPLE : distance entre 2 groupes = MIN des distances entre leurs points.
On fusionne à chaque étape les 2 groupes les plus proches, jusqu'à 1 seul groupe.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

FIG = os.path.join(os.path.dirname(__file__), "..", "rapport", "figures")
os.makedirs(FIG, exist_ok=True)

POINTS = ["A", "B", "C", "D", "E"]
# distances initiales (paires)
D0 = {
    ("A", "B"): 1, ("A", "C"): 3, ("A", "D"): 2, ("A", "E"): 4,
    ("B", "C"): 3, ("B", "D"): 2, ("B", "E"): 3,
    ("C", "D"): 1, ("C", "E"): 3,
    ("D", "E"): 5,
}


def dist(a, b):
    return D0[(a, b)] if (a, b) in D0 else D0[(b, a)]


def lien_simple(g1, g2):
    """min des distances entre tout point de g1 et tout point de g2."""
    return min(dist(x, y) for x in g1 for y in g2)


def main():
    print("=" * 70)
    print("CLUSTERING HIÉRARCHIQUE ASCENDANT — LIEN SIMPLE")
    print("=" * 70)

    # chaque point dans son propre cluster, représenté par un tuple de membres
    clusters = [(p,) for p in POINTS]
    etape = 0
    while len(clusters) > 1:
        etape += 1
        # trouver la paire de clusters la plus proche (lien simple)
        meilleure = (float("inf"), None, None)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = lien_simple(clusters[i], clusters[j])
                if d < meilleure[0]:
                    meilleure = (d, i, j)
        d, i, j = meilleure
        ci, cj = clusters[i], clusters[j]
        fusion = ci + cj
        print(f"\nÉtape {etape} : plus proches = {ci} et {cj}  (distance lien simple = {d})")
        print(f"          -> fusion en {fusion}")
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [fusion]
        # afficher la matrice mise à jour (lien simple) entre clusters restants
        if len(clusters) > 1:
            print("   distances mises à jour :")
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    print(f"      d({clusters[a]},{clusters[b]}) = "
                          f"{lien_simple(clusters[a], clusters[b])}")

    print(f"\nRésultat : un seul groupe {clusters[0]}")
    print("Ordre des fusions (dendrogramme) : {A,B}@1, {C,D}@1, {AB,CD}@2, {ABCD,E}@3")

    # --- dendrogramme avec scipy (vérification + figure) ---
    # matrice condensée au format scipy : ordre AB, AC, AD, AE, BC, BD, BE, CD, CE, DE
    condensee = np.array([
        dist("A", "B"), dist("A", "C"), dist("A", "D"), dist("A", "E"),
        dist("B", "C"), dist("B", "D"), dist("B", "E"),
        dist("C", "D"), dist("C", "E"),
        dist("D", "E"),
    ], dtype=float)
    Z = linkage(condensee, method="single")
    print("\n[scipy] matrice de liaison (cluster1, cluster2, distance, taille) :")
    print(Z)

    plt.figure(figsize=(7, 4.5))
    dendrogram(Z, labels=POINTS)
    plt.title("Dendrogramme — lien simple")
    plt.ylabel("distance de fusion")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "hierarchique_dendrogramme.png"), dpi=130)
    print("[figure] rapport/figures/hierarchique_dendrogramme.png enregistrée")


if __name__ == "__main__":
    main()
