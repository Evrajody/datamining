"""
RÈGLES D'ASSOCIATION — algorithme APRIORI (implémenté à la main)
================================================================
Vocabulaire :
  - transaction : un panier (ensemble d'articles), ex. {pain, lait, vin}
  - itemset     : un ensemble d'articles
  - support(X)  = proportion de transactions contenant X
  - règle X -> Y : "si X alors Y"
        confiance(X->Y) = support(X ∪ Y) / support(X)
        lift(X->Y)      = confiance / support(Y)   (>1 = corrélation positive)

APRIORI repose sur une idée simple : si un itemset est fréquent, tous ses
sous-ensembles le sont aussi. Donc on construit les itemsets fréquents de
taille croissante, en n'élargissant que ceux qui ont passé le seuil de support.
"""

from itertools import combinations


def _support(itemset, transactions):
    """proportion de transactions qui contiennent tous les articles de itemset."""
    s = set(itemset)
    n = sum(1 for t in transactions if s.issubset(t))
    return n / len(transactions)


def itemsets_frequents(transactions, min_support):
    """Renvoie {frozenset: support} pour tous les itemsets fréquents."""
    transactions = [set(t) for t in transactions]
    # niveau 1 : articles individuels fréquents
    articles = sorted({a for t in transactions for a in t})
    courant = []
    frequents = {}
    for a in articles:
        sup = _support([a], transactions)
        if sup >= min_support:
            courant.append((a,))
            frequents[frozenset([a])] = sup

    k = 2
    while courant:
        # génération des candidats de taille k par union des fréquents précédents
        candidats = set()
        items = sorted({a for c in courant for a in c})
        for combo in combinations(items, k):
            # élagage Apriori : tous les sous-ensembles (k-1) doivent être fréquents
            if all(frozenset(sub) in frequents for sub in combinations(combo, k - 1)):
                candidats.add(combo)
        # filtrage par support
        courant = []
        for cand in sorted(candidats):
            sup = _support(cand, transactions)
            if sup >= min_support:
                courant.append(cand)
                frequents[frozenset(cand)] = sup
        k += 1
    return frequents


def regles_association(transactions, min_support=0.3, min_confidence=0.6):
    """Génère les règles X -> Y dépassant les seuils de support et de confiance."""
    transactions = [set(t) for t in transactions]
    frequents = itemsets_frequents(transactions, min_support)
    regles = []
    for itemset, sup in frequents.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        # toutes les façons de couper itemset en antécédent X et conséquent Y
        for r in range(1, len(items)):
            for X in combinations(items, r):
                X = frozenset(X)
                Y = itemset - X
                conf = sup / frequents[X]
                if conf >= min_confidence:
                    lift = conf / frequents[Y]
                    regles.append(
                        {
                            "antecedent": set(X),
                            "consequent": set(Y),
                            "support": round(sup, 3),
                            "confidence": round(conf, 3),
                            "lift": round(lift, 3),
                        }
                    )
    return sorted(regles, key=lambda r: (r["confidence"], r["lift"]), reverse=True)


if __name__ == "__main__":
    # petit exemple "panier de la ménagère"
    paniers = [
        {"pain", "lait", "vin"},
        {"pain", "lait"},
        {"vin", "fromage"},
        {"pain", "vin", "fromage"},
        {"pain", "lait", "vin", "fromage"},
    ]
    print("Itemsets fréquents (min_support=0.4) :")
    for it, sup in itemsets_frequents(paniers, 0.4).items():
        print(f"  {set(it)} -> support={sup:.2f}")
    print("\nRègles (min_conf=0.6) :")
    for r in regles_association(paniers, 0.4, 0.6):
        print(f"  {r['antecedent']} -> {r['consequent']} "
              f"(supp={r['support']}, conf={r['confidence']}, lift={r['lift']})")
