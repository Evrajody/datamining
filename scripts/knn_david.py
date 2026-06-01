"""
CHAPITRE 4 — CLASSIFICATION : méthode des K plus proches voisins (K-NN)
=======================================================================
EXERCICE : classer le client "David" dans la classe N ou O, avec K = 3
           et la distance euclidienne (vote simple ET vote pondéré).

Données (Age, Salaire, Nb cartes de crédit, classe "Loyal") :
  Jean     35   35000  3  N
  Rachèle  22   50000  2  O
  Anne     63  200000  1  N
  Thomas   59  170000  1  N
  Nathalie 25   40000  4  O
  David    37   50000  2  ?   <-- à classer

On code la distance euclidienne à la main, puis on applique :
  - vote simple        : la classe majoritaire parmi les K voisins
  - vote pondéré       : chaque voisin a un poids = 1 / distance
La "confiance" = votes gagnants / total des votes.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "rapport", "figures")
os.makedirs(FIG, exist_ok=True)

# (nom, age, salaire, nb_cartes, classe)
APPRENTISSAGE = [
    ("Jean", 35, 35000, 3, "N"),
    ("Rachèle", 22, 50000, 2, "O"),
    ("Anne", 63, 200000, 1, "N"),
    ("Thomas", 59, 170000, 1, "N"),
    ("Nathalie", 25, 40000, 4, "O"),
]
DAVID = (37, 50000, 2)
K = 3


def distance_euclidienne(a, b) -> float:
    """racine( somme (a_i - b_i)^2 ) sur toutes les dimensions."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5


def knn(point, donnees, k):
    # 1) distance de 'point' à chaque individu connu
    distances = []
    for nom, age, sal, cartes, classe in donnees:
        d = distance_euclidienne(point, (age, sal, cartes))
        distances.append((nom, classe, d))
    # 2) tri par distance croissante, on garde les k plus proches
    distances.sort(key=lambda t: t[2])
    voisins = distances[:k]
    return distances, voisins


def vote_simple(voisins):
    scores = {}
    for _, classe, _ in voisins:
        scores[classe] = scores.get(classe, 0) + 1
    gagnant = max(scores, key=scores.get)
    confiance = scores[gagnant] / sum(scores.values())
    return gagnant, scores, confiance


def vote_pondere(voisins):
    # poids = 1/distance : plus un voisin est proche, plus il pèse lourd
    scores = {}
    for _, classe, d in voisins:
        scores[classe] = scores.get(classe, 0.0) + 1.0 / d
    gagnant = max(scores, key=scores.get)
    confiance = scores[gagnant] / sum(scores.values())
    return gagnant, scores, confiance


def normaliser_min_max(donnees, point):
    """Ramène chaque attribut dans [0,1] avec les min/max de l'apprentissage.
    Renvoie (donnees_normalisees, point_normalise)."""
    # min et max de chaque colonne (age, salaire, cartes) sur l'apprentissage
    colonnes = list(zip(*[(a, s, c) for _, a, s, c, _ in donnees]))
    mins = [min(col) for col in colonnes]
    maxs = [max(col) for col in colonnes]

    def norm(triplet):
        return tuple((v - mn) / (mx - mn) for v, mn, mx in zip(triplet, mins, maxs))

    dn = [(nom, *norm((a, s, c)), cl) for nom, a, s, c, cl in donnees]
    pn = norm(point)
    return dn, pn, mins, maxs


def main():
    print("=" * 70)
    print(f"K-NN — classer David {DAVID} avec K={K} (distance euclidienne)")
    print("=" * 70)

    distances, voisins = knn(DAVID, APPRENTISSAGE, K)

    print("\nDistances de David à chaque individu (détail du calcul) :")
    for nom, age, sal, cartes, classe in APPRENTISSAGE:
        d = distance_euclidienne(DAVID, (age, sal, cartes))
        print(
            f"  d(David,{nom:8s}) = racine(({DAVID[0]}-{age})² + "
            f"({DAVID[1]}-{sal})² + ({DAVID[2]}-{cartes})²) = {d:12.3f}  [{classe}]"
        )

    print(f"\nLes K={K} plus proches voisins :")
    for nom, classe, d in voisins:
        print(f"  {nom:8s}  classe={classe}  distance={d:.3f}")

    g1, s1, c1 = vote_simple(voisins)
    print(f"\nVOTE SIMPLE     : {s1}  -> classe = {g1}  (confiance {c1:.0%})")

    g2, s2, c2 = vote_pondere(voisins)
    s2r = {k: round(v, 6) for k, v in s2.items()}
    print(f"VOTE PONDÉRÉ    : {s2r}  -> classe = {g2}  (confiance {c2:.0%})")

    print(
        "\nConclusion : David est classé 'O' (Loyal=O) dans les deux votes.\n"
        "Remarque importante du cours : le salaire (≈ dizaines de milliers) écrase\n"
        "l'âge et le nombre de cartes. D'où la recommandation de NORMALISER avant\n"
        "un K-NN pour que chaque attribut compte autant."
    )

    # ----------------------------------------------------------------
    # VERSION NORMALISÉE (min-max) — recommandée par le cours
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("VERSION NORMALISÉE (min-max) : chaque attribut compte autant")
    print("-" * 70)
    dn, pn, mins, maxs = normaliser_min_max(APPRENTISSAGE, DAVID)
    print(f"min par colonne = {mins}, max = {maxs}")
    print(f"David normalisé = ({pn[0]:.3f}, {pn[1]:.3f}, {pn[2]:.3f})")
    dist_n = []
    for nom, a, s, c, classe in dn:
        d = distance_euclidienne(pn, (a, s, c))
        dist_n.append((nom, classe, d))
        print(f"  d(David,{nom:8s}) = {d:.3f}   [{classe}]   "
              f"normalisé=({a:.3f},{s:.3f},{c:.3f})")
    dist_n.sort(key=lambda t: t[2])
    voisins_n = dist_n[:K]
    gn, sn, cn = vote_simple(voisins_n)
    print(f"  3 plus proches normalisés : {[(n, round(d,3)) for n,_,d in voisins_n]}")
    print(f"  -> vote simple = {gn} (confiance {cn:.0%})")
    print("  Remarque : après normalisation, Jean (N) devient le PLUS proche,\n"
          "  mais 'O' l'emporte quand même 2 voix contre 1.")

    # Vérification avec scikit-learn
    try:
        from sklearn.neighbors import KNeighborsClassifier

        X = [[a, s, c] for _, a, s, c, _ in APPRENTISSAGE]
        y = [cl for *_, cl in APPRENTISSAGE]
        clf = KNeighborsClassifier(n_neighbors=K).fit(X, y)
        print(f"\n[sklearn] prédiction = {clf.predict([list(DAVID)])[0]}")
    except Exception as e:  # pragma: no cover
        print("sklearn indisponible:", e)

    # Figure : barres des distances (échelle log car salaire domine)
    noms = [t[0] for t in distances]
    ds = [t[2] for t in distances]
    cols = ["#55A868" if t[1] == "O" else "#C44E52" for t in distances]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(noms, ds, color=cols)
    ax.set_yscale("log")
    ax.set_ylabel("distance à David (log)")
    ax.set_title("Distances de David — vert=O, rouge=N (3 plus proches = O,O,N)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "knn_distances.png"), dpi=130)
    print("[figure] rapport/figures/knn_distances.png enregistrée")


if __name__ == "__main__":
    main()
