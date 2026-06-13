"""
ANALYSE DE SENTIMENT (français) — modèle DistilCamemBERT
========================================================
Classe des textes français en sentiment, via le modèle
« cmarkea/distilcamembert-base-sentiment » (sortie : 1 à 5 étoiles), puis
ramène l'étoile à une polarité : négatif / neutre / positif.

Usage prévu : scorer les posts/commentaires récupérés par le scraper Facebook
(fichiers JSON de off-dataset/), ou n'importe quelle liste de phrases.

PRÉREQUIS : transformers + un backend PyTorch.
    uv add transformers torch

EN LIGNE DE COMMANDE :
    # démonstration sur quelques phrases
    uv run python projet/transformers_test.py
    # scorer le champ "text" d'un JSON de posts -> écrit *_sentiment.json
    uv run python projet/transformers_test.py off-dataset/fb_posts_nintendo.json

EN MODULE :
    from transformers_test import analyser, analyser_lot
    analyser("Le service était correct mais l'attente interminable.")
    # {'texte': '...', 'etoiles': 2, 'polarite': 'négatif', 'score': 0.51}
"""

import json
import re
import sys
from functools import lru_cache

MODELE = "cmarkea/distilcamembert-base-sentiment"


@lru_cache(maxsize=1)
def _analyseur():
    """Construit le pipeline une seule fois (chargement paresseux + mis en cache)."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model=MODELE,
        truncation=True,
        max_length=512,
    )


def vers_polarite(label):
    """Label « N star(s) » -> 'négatif' (1-2), 'neutre' (3) ou 'positif' (4-5)."""
    m = re.search(r"\d", str(label))
    etoiles = int(m.group()) if m else 3
    if etoiles <= 2:
        return "négatif"
    if etoiles == 3:
        return "neutre"
    return "positif"


def _formater(texte, brut):
    """Sortie brute du pipeline -> dict propre et homogène."""
    m = re.search(r"\d", brut["label"])
    return {
        "texte": texte,
        "etoiles": int(m.group()) if m else None,
        "polarite": vers_polarite(brut["label"]),
        "score": round(float(brut["score"]), 4),
    }


def analyser(texte):
    """Analyse une seule phrase -> dict {texte, etoiles, polarite, score}."""
    brut = _analyseur()(texte)[0]
    return _formater(texte, brut)


def analyser_lot(textes, batch_size=16):
    """Analyse une liste de phrases (traitement par lots) -> liste de dicts."""
    textes = [t for t in textes if isinstance(t, str) and t.strip()]
    if not textes:
        return []
    bruts = _analyseur()(textes, batch_size=batch_size)
    return [_formater(t, b) for t, b in zip(textes, bruts)]


def analyser_json(chemin):
    """Score le champ « text » de chaque post d'un JSON et renvoie la liste enrichie."""
    with open(chemin, encoding="utf-8") as f:
        posts = json.load(f)
    textes = [p.get("text", "") for p in posts]
    scores = {s["texte"]: s for s in analyser_lot(textes)}
    for p in posts:
        s = scores.get(p.get("text", ""))
        p["polarite"] = s["polarite"] if s else None
        p["sentiment_etoiles"] = s["etoiles"] if s else None
        p["sentiment_score"] = s["score"] if s else None
    return posts


def _demo():
    phrases = [
        "Le service était correct mais l'attente interminable.",
        "Une expérience absolument géniale, je recommande vivement !",
        "C'était sans plus, ni bon ni mauvais.",
        "Produit défectueux et SAV catastrophique, à éviter.",
    ]
    for r in analyser_lot(phrases):
        print(f"  [{r['polarite']:<7}] {r['etoiles']}★ "
              f"({r['score']:.2f})  {r['texte']}")


def main():
    if len(sys.argv) > 1:                       # un chemin JSON est fourni
        chemin = sys.argv[1]
        posts = analyser_json(chemin)
        sortie = re.sub(r"\.json$", "_sentiment.json", chemin)
        with open(sortie, "w", encoding="utf-8") as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        n = sum(1 for p in posts if p.get("polarite"))
        print(f"✅  {n} texte(s) scoré(s) -> {sortie}")
    else:                                       # sinon : démonstration
        print("Démonstration (aucun fichier fourni) :")
        _demo()


if __name__ == "__main__":
    main()
