#!/usr/bin/env bash
# Régénère les figures (via les scripts) puis compile le PDF avec tectonic.
set -e
cd "$(dirname "$0")/.."

echo ">> Génération des figures (scripts Python)…"
for s in chap2_normalisation knn_david gini_arbre gain_information \
         kmeans_1d hierarchique_lien_simple performance_classification; do
  uv run python "scripts/$s.py" >/dev/null
done

echo ">> Compilation LaTeX (tectonic)…"
tectonic rapport/main.tex

echo ">> Terminé : rapport/main.pdf"
