"""
CHAPITRE 4 / TP — ARBRE DE DÉCISION par le GAIN D'INFORMATION (ID3 / C4.5)
==========================================================================
Données (achat d'ordinateur) — 14 individus :
  age{<=30, 31..40, >40}, revenue{élevé, moyen, bas}, étudiant{oui, non},
  Etat_civil{célibataire, marié}, classe = Achète_ordinateur{oui, non}

RAPPELS (cours), logarithme en base 2 :
  Entropie(t) = - somme_j p(j|t) * log2 p(j|t)        (0 = pur, 1 = 50/50)
  Gain(A)     = Entropie(parent) - somme_i (n_i/n) * Entropie(enfant_i)
  -> on choisit l'attribut de PLUS GRAND gain (réduction d'entropie maximale).
"""

import os
from math import log2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = os.path.join(os.path.dirname(__file__), "..", "rapport", "figures")
os.makedirs(FIG, exist_ok=True)

LIGNES = [
    ("<=30", "élevé", "non", "célibataire", "non"),
    ("<=30", "élevé", "non", "marié", "non"),
    ("31..40", "élevé", "non", "célibataire", "oui"),
    (">40", "moyen", "non", "célibataire", "oui"),
    (">40", "bas", "oui", "célibataire", "oui"),
    (">40", "bas", "oui", "marié", "non"),
    ("31..40", "bas", "oui", "marié", "oui"),
    ("<=30", "moyen", "non", "célibataire", "non"),
    ("<=30", "bas", "oui", "célibataire", "oui"),
    (">40", "moyen", "oui", "célibataire", "oui"),
    ("<=30", "moyen", "oui", "marié", "oui"),
    ("31..40", "moyen", "non", "marié", "oui"),
    ("31..40", "élevé", "oui", "célibataire", "oui"),
    (">40", "moyen", "non", "marié", "non"),
]
COLS = ["age", "revenue", "etudiant", "etat_civil", "achete"]
DATA = pd.DataFrame(LIGNES, columns=COLS)
CIBLE = "achete"


def entropie(serie) -> float:
    """- somme p*log2(p)."""
    n = len(serie)
    if n == 0:
        return 0.0
    h = 0.0
    for c in serie.unique():
        p = (serie == c).sum() / n
        if p > 0:
            h -= p * log2(p)
    return h


def gain(df, attr) -> tuple[float, float]:
    base = entropie(df[CIBLE])
    reste = 0.0
    for _, g in df.groupby(attr):
        reste += len(g) / len(df) * entropie(g[CIBLE])
    return base - reste, reste


def construire_id3(df, attributs, profondeur=0):
    """Arbre ID3 récursif. Renvoie soit une classe (feuille) soit un dict."""
    indent = "   " * profondeur
    classes = df[CIBLE].unique()
    if len(classes) == 1:  # nœud pur
        print(f"{indent}-> feuille : {classes[0]}")
        return classes[0]
    if not attributs:  # plus d'attribut : classe majoritaire
        maj = df[CIBLE].mode()[0]
        print(f"{indent}-> feuille (majorité) : {maj}")
        return maj
    # choisir l'attribut de plus grand gain
    gains = {a: gain(df, a)[0] for a in attributs}
    meilleur = max(gains, key=gains.get)
    print(f"{indent}NŒUD : on teste « {meilleur} »  "
          f"(gains: {{{', '.join(f'{a}:{g:.3f}' for a, g in gains.items())}}})")
    arbre = {meilleur: {}}
    for val, sous in df.groupby(meilleur):
        print(f"{indent}  {meilleur} = {val} :")
        reste = [a for a in attributs if a != meilleur]
        arbre[meilleur][val] = construire_id3(sous, reste, profondeur + 2)
    return arbre


def main():
    print("=" * 70)
    print("ARBRE DE DÉCISION — GAIN D'INFORMATION (ID3)")
    print("=" * 70)
    print(DATA.to_string(index=False))

    base = entropie(DATA[CIBLE])
    no, yes = (DATA[CIBLE] == "non").sum(), (DATA[CIBLE] == "oui").sum()
    print(f"\nEntropie(racine) = -({yes}/14)log2({yes}/14) -({no}/14)log2({no}/14) "
          f"= {base:.3f}")

    print("\nGain de chaque attribut au 1er niveau :")
    for a in ["age", "revenue", "etudiant", "etat_civil"]:
        g, reste = gain(DATA, a)
        print(f"   Gain({a:11s}) = {base:.3f} - {reste:.3f} = {g:.3f}")
    print("   => 'age' a le plus grand gain : c'est la RACINE.\n")

    print("CONSTRUCTION RÉCURSIVE :")
    construire_id3(DATA, ["age", "revenue", "etudiant", "etat_civil"])

    print("\nARBRE FINAL :")
    print("  age ?")
    print("   ├─ <=30   -> étudiant ? oui->oui , non->non")
    print("   ├─ 31..40 -> oui   (feuille pure)")
    print("   └─ >40    -> etat_civil ? célibataire->oui , marié->non")

    # --- Vérification scikit-learn (entropy) + figure ---
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

    enc = pd.get_dummies(DATA.drop(columns=[CIBLE]))
    y = DATA[CIBLE]
    model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(enc, y)
    print("\n[sklearn, criterion=entropy] règles :")
    print(export_text(model, feature_names=list(enc.columns)))

    plt.figure(figsize=(13, 7))
    plot_tree(model, feature_names=list(enc.columns),
              class_names=model.classes_, filled=True, rounded=True, fontsize=8)
    plt.title("Arbre de décision — Gain d'information / entropie (scikit-learn)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "gain_information_arbre.png"), dpi=130)
    print("[figure] rapport/figures/gain_information_arbre.png enregistrée")


if __name__ == "__main__":
    main()
