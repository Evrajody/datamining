# Projet de Data Mining — Master 2 (ENEAM / UAC)

Application qui réalise les **trois tâches** classiques du data mining sur un jeu
de données au choix (upload CSV ou jeu de démonstration), avec mesures de
performance et visualisations :

| Tâche | Algorithmes |
|-------|-------------|
| **Segmentation** | K-means, Agglomératif (CAH) |
| **Classification** | Arbre de décision, k plus proches voisins (k-NN), Forêt aléatoire |
| **Recherche d'associations** | Apriori *(implémenté à la main)* |

Le projet contient aussi le **rapport LaTeX** (cours + TP + projet) et les
**scripts** qui régénèrent ses figures.

---

## 🧩 Les 3 versions de l'interface

La même logique de calcul (scikit-learn + `apriori.py`) est partagée par trois
interfaces. Choisis celle qui te convient — **elles font le même travail** :

| Version | Fichier | Techno | Pour qui |
|---------|---------|--------|----------|
| **Web** ⭐ | `projet/app_web.py` | [NiceGUI](https://nicegui.io) | Rendu moderne, graphes interactifs (ECharts), 100 % Python. Recommandée. |
| **Desktop** | `projet/app_tk.py` | [CustomTkinter](https://customtkinter.tomschimansky.com) | Fenêtre native, sans navigateur. |
| **Originale** | `projet/app.py` | [Streamlit](https://streamlit.io) | Version historique, simple et rapide. |

> Le cœur de calcul est dans `projet/apriori.py` (Apriori) et réutilisé tel quel
> par les trois interfaces. Changer d'UI ne change rien aux résultats.

---

## ⚙️ Prérequis

- **Python ≥ 3.12**
- Au choix : [`uv`](https://docs.astral.sh/uv/) *(recommandé)* **ou** `pip`

---

## 🚀 Lancement avec `uv` (recommandé)

`uv` lit `pyproject.toml` / `uv.lock` et installe tout automatiquement dans un
environnement isolé. **Aucune installation manuelle de dépendances.**

```bash
# 1. Se placer à la racine du projet
cd datamining

# 2. Synchroniser l'environnement (une seule fois)
uv sync
```

Puis lance la version voulue :

```bash
# Version Web (NiceGUI)  → ouvre http://localhost:8080
uv run python projet/app_web.py

# Version Desktop (CustomTkinter)  → fenêtre native
uv run python projet/app_tk.py

# Version originale (Streamlit)  → ouvre http://localhost:8501
uv run streamlit run projet/app.py
```

> 💡 **Fenêtre desktop pour la version Web** : dans `projet/app_web.py`, dernière
> ligne, passe `native=False` → `native=True` (nécessite `uv add pywebview`).

---

## 🐍 Lancement avec `pip` (sans uv)

Crée un environnement virtuel puis installe **uniquement** ce dont la version
choisie a besoin.

```bash
# 1. Se placer à la racine et créer l'environnement
cd datamining
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 2. Dépendances communes (toujours nécessaires)
pip install pandas scikit-learn matplotlib numpy
```

Ensuite, selon la version :

### ▸ Version Web (NiceGUI)
```bash
pip install nicegui
python projet/app_web.py
# → http://localhost:8080
```

### ▸ Version Desktop (CustomTkinter)
```bash
pip install customtkinter
python projet/app_tk.py
```

### ▸ Version originale (Streamlit)
```bash
pip install streamlit
streamlit run projet/app.py
# → http://localhost:8501
```

> Tout installer d'un coup : `pip install pandas scikit-learn matplotlib numpy nicegui customtkinter streamlit`

---

## 🧪 Utilisation (identique pour les 3 versions)

1. **Choisir la tâche** (segmentation / classification / associations).
2. **Choisir l'algorithme** proposé pour cette tâche.
3. **Fournir les données** : *upload d'un CSV* **ou** *« Jeu de démonstration »*.
4. **Régler les paramètres** (attribut à prédire, split, profondeur, k, seuils…).
5. **Lancer** → résultats : métriques (accuracy, précision, rappel, F1,
   silhouette…), matrice de confusion, scatter PCA, arbre, règles support /
   confiance / lift.

> ⚠️ `final.csv` (à la racine) n'est pas adapté à ces algorithmes : utilise les
> **boutons de démonstration**, qui chargent un jeu pensé pour chaque tâche.

---

## 📄 Rapport LaTeX

Le rapport (cours + TP + projet) se compile avec [tectonic](https://tectonic-typesetting.github.io).

```bash
# Régénère les figures (scripts Python) puis compile le PDF
bash rapport/build.sh
# → rapport/main.pdf
```

Compilation manuelle si les figures sont déjà à jour :

```bash
tectonic rapport/main.tex
```

---

## 🗂️ Structure du dépôt

```
datamining/
├── projet/
│   ├── app_web.py      # Interface Web (NiceGUI)         ⭐ recommandée
│   ├── app_tk.py       # Interface Desktop (CustomTkinter)
│   ├── app.py          # Interface Streamlit (originale)
│   └── apriori.py      # Algorithme Apriori (fait main)
├── scripts/            # Génération des figures du rapport
├── rapport/            # Sources LaTeX + build.sh + main.pdf
├── cours/              # Supports de cours (PDF)
├── pyproject.toml      # Dépendances (uv)
└── README.md
```
