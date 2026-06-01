"""
CHAPITRE 4 — ARBRE DE DÉCISION par l'INDICE DE GINI (CART)
==========================================================
Données d'apprentissage (assurance auto) :
  ID  Age  Type_voit   Risque
  0   23   Familiale   Élevé
  1   17   Sport       Élevé
  2   43   Sport       Élevé
  3   68   Familiale   Faible
  4   32   Camion      Faible
  5   20   Familiale   Élevé

RAPPELS (cours) :
  Gini(t) = 1 - somme_j p(j|t)^2          (0 = nœud pur, plus bas = plus pur)
  Gini_split = somme_i (n_i / n) * Gini(i)   (moyenne pondérée des enfants)
  -> on choisit le découpage qui MINIMISE Gini_split.

On code tout à la main pour voir chaque calcul, puis on vérifie avec scikit-learn.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = os.path.join(os.path.dirname(__file__), "..", "rapport", "figures")
os.makedirs(FIG, exist_ok=True)

DATA = pd.DataFrame(
    {
        "Age": [23, 17, 43, 68, 32, 20],
        "Type_voit": ["Familiale", "Sport", "Sport", "Familiale", "Camion", "Familiale"],
        "Risque": ["Élevé", "Élevé", "Élevé", "Faible", "Faible", "Élevé"],
    }
)


def gini(labels) -> float:
    """1 - somme des p^2. labels = liste/série de classes."""
    n = len(labels)
    if n == 0:
        return 0.0
    g = 1.0
    for c in set(labels):
        p = sum(1 for x in labels if x == c) / n
        g -= p**2
    return g


def gini_split(groupes) -> float:
    """Moyenne pondérée des Gini des groupes (enfants)."""
    n = sum(len(g) for g in groupes)
    return sum(len(g) / n * gini(g) for g in groupes)


def evaluer_attribut_nominal(df, attr, cible):
    """Découpage multi-branche : un enfant par valeur de l'attribut."""
    groupes = [g[cible].tolist() for _, g in df.groupby(attr)]
    return gini_split(groupes), {v: g[cible].tolist() for v, g in df.groupby(attr)}


def evaluer_attribut_continu(df, attr, cible):
    """Découpage binaire (attr < seuil). On teste tous les milieux de valeurs."""
    vals = sorted(df[attr].unique())
    meilleurs = (1.0, None, None)
    details = []
    for i in range(len(vals) - 1):
        seuil = (vals[i] + vals[i + 1]) / 2
        gauche = df[df[attr] < seuil][cible].tolist()
        droite = df[df[attr] >= seuil][cible].tolist()
        gs = gini_split([gauche, droite])
        details.append((seuil, gauche, droite, gs))
        if gs < meilleurs[0]:
            meilleurs = (gs, seuil, (gauche, droite))
    return meilleurs, details


def main():
    print("=" * 70)
    print("ARBRE DE DÉCISION — INDICE DE GINI")
    print("=" * 70)
    print(DATA.to_string(index=False))

    racine = DATA["Risque"].tolist()
    print(f"\nGini(racine) = {gini(racine):.4f}  "
          f"(Élevé={racine.count('Élevé')}, Faible={racine.count('Faible')})")

    # --- Attribut Type_voit (nominal) ---
    gs_type, groupes = evaluer_attribut_nominal(DATA, "Type_voit", "Risque")
    print("\n[Type_voit] (multi-branche)")
    for v, lab in groupes.items():
        print(f"   {v:10s} {lab}  Gini={gini(lab):.4f}")
    print(f"   => Gini_split(Type_voit) = {gs_type:.4f}")

    # --- Attribut Age (continu) ---
    print("\n[Age] (binaire, test de tous les seuils)")
    (gs_age, seuil, _), details = evaluer_attribut_continu(DATA, "Age", "Risque")
    for s, gch, dr, gsv in details:
        print(f"   Age < {s:5.1f} : gauche={gch} droite={dr}  Gini_split={gsv:.4f}")
    print(f"   => meilleur seuil Age < {seuil}  Gini_split = {gs_age:.4f}")

    print("\nComparaison du 1er découpage :")
    print(f"   Type_voit -> {gs_type:.4f}   |   Age<{seuil} -> {gs_age:.4f}")
    print("   (égalité ! scikit-learn choisit Age<27.5, on suit ce choix)\n")

    print("ARBRE OBTENU :")
    print("  Age < 27.5 ?")
    print("   ├─ OUI -> Élevé  (feuille pure : ages 17,20,23 tous Élevé)")
    print("   └─ NON -> Type_voit ?")
    print("            ├─ Sport     -> Élevé")
    print("            ├─ Familiale -> Faible")
    print("            └─ Camion    -> Faible")

    # --- Vérification scikit-learn + figure ---
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

    enc = pd.get_dummies(DATA, columns=["Type_voit"])
    X = enc.drop("Risque", axis=1)
    y = enc["Risque"]
    model = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X, y)
    print("\n[sklearn] règles de l'arbre :")
    print(export_text(model, feature_names=list(X.columns)))

    plt.figure(figsize=(11, 6))
    plot_tree(model, feature_names=list(X.columns),
              class_names=model.classes_, filled=True, rounded=True, fontsize=9)
    plt.title("Arbre de décision — Indice de Gini (scikit-learn)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "gini_arbre.png"), dpi=130)
    print("[figure] rapport/figures/gini_arbre.png enregistrée")


if __name__ == "__main__":
    main()
