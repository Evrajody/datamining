"""
CHAPITRE 5 — SEGMENTATION : algorithme K-MEANS (centres mobiles)
================================================================
EXERCICE : C = {2, 3, 4, 10, 11, 12, 20, 25, 30}
           k = 2, centres initiaux  mu1 = 2 , mu2 = 4

PRINCIPE (cours) :
  1. AFFECTATION  : chaque point va au centre le plus proche.
  2. MISE À JOUR  : chaque centre devient la moyenne des points de son cluster.
  On répète jusqu'à ce que les centres ne bougent plus (convergence).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "rapport", "figures")
os.makedirs(FIG, exist_ok=True)

C = [2, 3, 4, 10, 11, 12, 20, 25, 30]


def kmeans_1d(points, mu1, mu2, max_iter=20):
    historique = []
    for it in range(1, max_iter + 1):
        # 1) AFFECTATION : point vers le centre le plus proche (|.|)
        g1, g2 = [], []
        for x in points:
            (g1 if abs(x - mu1) <= abs(x - mu2) else g2).append(x)
        # 2) MISE À JOUR : moyenne de chaque groupe
        n1 = sum(g1) / len(g1) if g1 else mu1
        n2 = sum(g2) / len(g2) if g2 else mu2
        historique.append((it, mu1, mu2, list(g1), list(g2), n1, n2))
        if n1 == mu1 and n2 == mu2:  # convergence
            break
        mu1, mu2 = n1, n2
    return historique


def main():
    print("=" * 70)
    print("K-MEANS — C = {2,3,4,10,11,12,20,25,30}, k=2, mu1=2, mu2=4")
    print("=" * 70)
    hist = kmeans_1d(C, 2.0, 4.0)

    for it, m1, m2, g1, g2, n1, n2 in hist:
        print(f"\nItération {it} : centres mu1={m1}, mu2={m2}")
        for x in C:
            d1, d2 = abs(x - m1), abs(x - m2)
            choix = "C1" if d1 <= d2 else "C2"
            print(
                f"   x={x:2d} : |{x}-{m1}|={d1:5.2f}  |{x}-{m2}|={d2:5.2f}  -> {choix}"
            )
        print(f"   C1 = {g1}  -> nouvelle moyenne = {n1:.3f}")
        print(f"   C2 = {g2}  -> nouvelle moyenne = {n2:.3f}")

    it, m1, m2, g1, g2, n1, n2 = hist[-1]
    print("\nCONVERGENCE :")
    print(f"   Cluster 1 = {g1}  (centre = {n1})")
    print(f"   Cluster 2 = {g2}  (centre = {n2})")

    # inertie intra-cluster = somme des carrés des écarts au centre
    J1 = sum((x - n1) ** 2 for x in g1)
    J2 = sum((x - n2) ** 2 for x in g2)
    print(f"   Inertie J1 = {J1}, J2 = {J2}, total J = {J1 + J2}")

    # Vérification scikit-learn
    try:
        import numpy as np
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=2, init=np.array([[2.0], [4.0]]), n_init=1).fit(
            np.array(C).reshape(-1, 1)
        )
        print(f"\n[sklearn] centres = {sorted(km.cluster_centers_.ravel().round(3))}")
    except Exception as e:  # pragma: no cover
        print("sklearn indisponible:", e)

    # Figure de la convergence (centres au fil des itérations)
    fig, ax = plt.subplots(figsize=(8, 4))
    its = [h[0] for h in hist]
    ax.plot(its, [h[5] for h in hist], "o-", label="centre C1", color="#4C72B0")
    ax.plot(its, [h[6] for h in hist], "s-", label="centre C2", color="#C44E52")
    ax.set_xlabel("itération")
    ax.set_ylabel("position du centre")
    ax.set_title("K-means : convergence des centres")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "kmeans_convergence.png"), dpi=130)

    # Figure des points colorés par cluster final
    fig2, ax2 = plt.subplots(figsize=(8, 2.4))
    for x in C:
        col = "#4C72B0" if x in g1 else "#C44E52"
        ax2.scatter(x, 0, s=180, color=col, zorder=3)
        ax2.annotate(
            str(x), (x, 0), textcoords="offset points", xytext=(0, 10), ha="center"
        )
    ax2.scatter(
        [n1, n2], [0, 0], marker="X", s=260, color="black", zorder=4, label="centres"
    )
    ax2.set_yticks([])
    ax2.set_title("Partition finale : C1 (bleu) vs C2 (rouge)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG, "kmeans_partition.png"), dpi=130)
    print("[figures] kmeans_convergence.png et kmeans_partition.png enregistrées")


if __name__ == "__main__":
    main()
