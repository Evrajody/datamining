"""
PROJET DE DATA MINING — Application graphique (Streamlit)
=========================================================
Réalise les 3 tâches demandées par le sujet :
   1) SEGMENTATION        (K-means, Agglomératif)
   2) CLASSIFICATION      (Arbre de décision, K-NN, Forêt aléatoire)
   3) RECHERCHE D'ASSOCIATIONS (Apriori)

Fonctionnalités (cahier des charges du projet) :
   - choisir la tâche
   - renseigner le dataset (upload CSV) OU utiliser un jeu de démonstration
   - choisir l'algorithme et ses paramètres
   - lancer l'analyse
   - afficher les résultats + mesures de performance
   - comparaison indicative avec WEKA

Lancement :  uv run streamlit run projet/app.py
"""

import io

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from apriori import regles_association

# --------------------------------------------------------------------------
# CONFIG & STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Mining Studio",
    page_icon=":material/database:",
    layout="centered",  # colonne centrée et plus étroite (moins "wide")
)

st.markdown(
    """
    <style>
    /* titre centré avec dégradé, bien visible et descendu */
    .big-title {font-size:3.4rem;font-weight:900;text-align:center;
      margin:2.2rem 0 .2rem 0;line-height:1.1;letter-spacing:.5px;
      background:linear-gradient(90deg,#5B8BD0,#8E44AD);
      -webkit-background-clip:text;background-clip:text;
      -webkit-text-fill-color:transparent;}
    .subtitle {text-align:center;font-size:1.15rem;font-weight:500;
      color:#8A8FA3;margin:0 0 .4rem 0;}
    /* on resserre la colonne centrale et on laisse de l'air en haut */
    .block-container{max-width:880px;padding-top:3rem;}
    /* cartes de métriques : fond clair + texte foncé, lisible quel que soit le thème */
    [data-testid="stMetric"]{
      background:#EEF2FA;border:1px solid #C9D6F0;border-radius:12px;padding:10px 14px;
      text-align:center;}
    [data-testid="stMetric"] *{color:#1F2A44 !important;}
    [data-testid="stMetricValue"]{font-weight:800;}
    [data-testid="stMetricLabel"] p{color:#3A4A6B !important;}
    [data-testid="stMetricLabel"]{justify-content:center;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# JEUX DE DÉMONSTRATION (pour que l'app marche sans fichier)
# --------------------------------------------------------------------------
@st.cache_data
def demo_segmentation():
    """Type 'cereals' : caractéristiques nutritionnelles."""
    rng = np.random.default_rng(0)
    g1 = rng.normal([110, 1, 8], [10, 0.5, 1.5], (25, 3))   # céréales légères
    g2 = rng.normal([150, 5, 2], [12, 1.0, 1.0], (25, 3))   # riches/sucrées
    g3 = rng.normal([90, 3, 14], [8, 0.8, 2.0], (25, 3))    # fibres
    X = np.vstack([g1, g2, g3])
    return pd.DataFrame(X, columns=["calories", "proteines", "fibres"]).round(1)


@st.cache_data
def demo_classification():
    """Type 'customer churn' : le client résilie-t-il ?"""
    rng = np.random.default_rng(1)
    n = 300
    anciennete = rng.integers(1, 72, n)
    minutes = rng.normal(180, 60, n).clip(0)
    appels_service = rng.poisson(1.5, n)
    forfait_intl = rng.integers(0, 2, n)
    # règle "cachée" : churn si peu d'ancienneté + bcp d'appels au service client
    proba = 1 / (1 + np.exp(-(0.06 * appels_service * 2 + 0.02 * (12 - anciennete) - 0.5)))
    churn = (rng.random(n) < proba).astype(int)
    return pd.DataFrame(
        {
            "anciennete": anciennete,
            "minutes_jour": minutes.round(1),
            "appels_service": appels_service,
            "forfait_intl": forfait_intl,
            "churn": np.where(churn == 1, "oui", "non"),
        }
    )


@st.cache_data
def demo_association():
    """Type 'panier' : transactions de supermarché (1 ligne = 1 panier)."""
    paniers = [
        "pain,lait,vin", "pain,lait", "vin,fromage", "pain,vin,fromage",
        "pain,lait,vin,fromage", "lait,fromage", "pain,vin", "vin,fromage,oeufs",
        "pain,lait,oeufs", "vin,fromage", "pain,vin,fromage,oeufs", "lait,oeufs",
        "pain,lait,vin", "vin,fromage", "pain,lait,fromage",
    ]
    return pd.DataFrame({"panier": paniers})


def telecharger(df, nom):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Télécharger les résultats (CSV)", buf.getvalue(),
                       file_name=nom, mime="text/csv", icon=":material/download:")


# --------------------------------------------------------------------------
# SIDEBAR — choix de la tâche et des données
# Icônes natives "Material Symbols" via la syntaxe :material/nom:
# --------------------------------------------------------------------------
# libellés des tâches associés à leur icône Material
TACHES = {
    "Segmentation": ":material/bubble_chart:",
    "Classification": ":material/account_tree:",
    "Recherche d'associations": ":material/shopping_cart:",
}

st.sidebar.markdown("### :material/database: Data Mining Studio")
tache = st.sidebar.radio(
    ":material/checklist: Choisir la tâche",
    list(TACHES),
    format_func=lambda t: f"{TACHES[t]}  {t}",  # affiche l'icône devant le libellé
)

st.sidebar.divider()
source = st.sidebar.radio(":material/folder_open: Source des données",
                          ["Jeu de démonstration", "Importer un CSV"])

fichier = None
if source == "Importer un CSV":
    fichier = st.sidebar.file_uploader("Fichier .csv", type=["csv"])
    sep = st.sidebar.selectbox("Séparateur", [",", ";", "\\t"], index=0)


def charger(demo_func):
    if source == "Importer un CSV" and fichier is not None:
        s = "\t" if sep == "\\t" else sep
        return pd.read_csv(fichier, sep=s)
    return demo_func()


st.markdown('<p class="big-title">Data Mining Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Master 2 IG/ID, ENEAM/UAC · segmentation · '
            'classification · règles d\'association</p>', unsafe_allow_html=True)
st.divider()


# ==========================================================================
# 1) SEGMENTATION
# ==========================================================================
if tache == "Segmentation":
    st.header(":material/bubble_chart: Segmentation (clustering)")
    df = charger(demo_segmentation)
    st.dataframe(df.head(), width="stretch")

    num_cols = df.select_dtypes("number").columns.tolist()
    cols = st.multiselect("Variables numériques à utiliser", num_cols, default=num_cols)
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithme", ["K-means", "Agglomératif (hiérarchique)"])
    k = c2.slider("Nombre de clusters K", 2, 8, 3)
    normaliser = c3.checkbox("Normaliser (z-score)", value=True)

    if st.button("Lancer la segmentation", type="primary",
                 icon=":material/play_arrow:", width="stretch") and cols:
        X = df[cols].to_numpy()
        if normaliser:
            X = StandardScaler().fit_transform(X)

        if algo == "K-means":
            model = KMeans(n_clusters=k, n_init=10, random_state=0)
        else:
            model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)

        # --- mesures de performance ---
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        m1, m2 = st.columns(2)
        m1.metric("Silhouette (−1→1, +haut = mieux)", f"{sil:.3f}")
        if algo == "K-means":
            m2.metric("Inertie intra-cluster", f"{model.inertia_:.1f}")
        else:
            m2.metric("Clusters formés", k)

        # --- projection 2D (PCA) ---
        proj = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X[:, :2]
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=40)
        if algo == "K-means":
            cen = PCA(n_components=2).fit(X).transform(model.cluster_centers_) \
                if X.shape[1] > 2 else model.cluster_centers_[:, :2]
            ax.scatter(cen[:, 0], cen[:, 1], marker="X", s=250, c="black", label="centres")
            ax.legend()
        ax.set_title(f"{algo} — {k} clusters (projection 2D)")
        st.pyplot(fig)

        res = df.copy()
        res["cluster"] = labels
        st.subheader("Taille des clusters")
        st.bar_chart(res["cluster"].value_counts().sort_index())
        st.dataframe(res, width="stretch")
        telecharger(res, "segmentation_resultats.csv")

        st.info("**Comparaison WEKA** : *Cluster → SimpleKMeans* (mêmes K et "
                "normalisation). WEKA affiche le `Within cluster sum of squared errors` "
                "(= inertie) ; il doit être du même ordre de grandeur.",
                icon=":material/science:")

    st.info("*Astuce du cours :* normaliser avant un clustering (les distances "
            "dépendent des échelles). Choisir K via la **silhouette** ou la méthode du "
            "**coude**.", icon=":material/lightbulb:")


# ==========================================================================
# 2) CLASSIFICATION
# ==========================================================================
elif tache == "Classification":
    st.header(":material/account_tree: Classification (supervisée)")
    df = charger(demo_classification)
    st.dataframe(df.head(), width="stretch")

    cible = st.selectbox("Variable cible (la classe à prédire)", df.columns,
                         index=len(df.columns) - 1)
    feats = [c for c in df.columns if c != cible]
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithme",
                        ["Arbre de décision", "K plus proches voisins", "Forêt aléatoire"])
    test_size = c2.slider("Part du jeu de test (%)", 10, 50, 33) / 100
    if algo == "Arbre de décision":
        param = c3.slider("Profondeur max", 1, 10, 4)
    elif algo == "K plus proches voisins":
        param = c3.slider("K (voisins)", 1, 15, 5)
    else:
        param = c3.slider("Nombre d'arbres", 10, 200, 100, step=10)

    if st.button("Entraîner & évaluer", type="primary",
                 icon=":material/play_arrow:", width="stretch"):
        X = pd.get_dummies(df[feats])           # encodage des variables nominales
        y = df[cible].astype(str)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=0, stratify=y
        )

        if algo == "Arbre de décision":
            model = DecisionTreeClassifier(max_depth=param, random_state=0)
        elif algo == "K plus proches voisins":
            # K-NN : on standardise (distances !)
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            model = KNeighborsClassifier(n_neighbors=param)
        else:
            model = RandomForestClassifier(n_estimators=param, random_state=0)
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)

        # --- mesures de performance ---
        classes = sorted(y.unique())
        avg = "binary" if len(classes) == 2 else "macro"
        pos = classes[-1] if len(classes) == 2 else None
        acc = accuracy_score(yte, yp)
        prec = precision_score(yte, yp, average=avg, pos_label=pos, zero_division=0)
        rec = recall_score(yte, yp, average=avg, pos_label=pos, zero_division=0)
        f1 = f1_score(yte, yp, average=avg, pos_label=pos, zero_division=0)

        a, b, c, d = st.columns(4)
        a.metric("Exactitude", f"{acc:.1%}")
        b.metric("Précision", f"{prec:.1%}")
        c.metric("Rappel", f"{rec:.1%}")
        d.metric("F1-score", f"{f1:.1%}")

        # --- matrice de confusion ---
        cm = confusion_matrix(yte, yp, labels=classes)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=45)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("prédit")
        ax.set_ylabel("réel")
        ax.set_title("Matrice de confusion")
        col_g, col_d = st.columns(2)
        col_g.pyplot(fig)

        # --- visualisation spécifique ---
        if algo == "Arbre de décision":
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            plot_tree(model, feature_names=list(X.columns), class_names=model.classes_,
                      filled=True, rounded=True, fontsize=7, max_depth=3)
            ax2.set_title("Arbre (3 premiers niveaux)")
            col_d.pyplot(fig2)
        elif algo == "Forêt aléatoire":
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
            col_d.bar_chart(imp)

        st.success("Modèle entraîné. Le jeu a été coupé en apprentissage / test "
                   f"({100*(1-test_size):.0f}% / {100*test_size:.0f}%).",
                   icon=":material/check_circle:")
        st.info("**Comparaison WEKA** : *Classify → trees/J48* (arbre), *lazy/IBk* "
                "(K-NN) ou *trees/RandomForest*. Comparer l'**accuracy** et la matrice de "
                "confusion de la sortie WEKA à celles ci-dessus.", icon=":material/science:")


# ==========================================================================
# 3) RECHERCHE D'ASSOCIATIONS
# ==========================================================================
else:
    st.header(":material/shopping_cart: Recherche d'associations (Apriori)")
    df = charger(demo_association)
    st.dataframe(df.head(), width="stretch")

    mode = st.radio(
        "Format des données",
        ["Colonne 'panier' (articles séparés par des virgules)",
         "Tableau (chaque colonne catégorielle = un item 'colonne=valeur')"],
    )
    col_panier = None
    if mode.startswith("Colonne"):
        defaut = "panier" if "panier" in df.columns else df.columns[0]
        col_panier = st.selectbox("Colonne contenant les paniers", df.columns,
                                  index=list(df.columns).index(defaut))
    c1, c2 = st.columns(2)
    min_sup = c1.slider("Support minimum", 0.05, 1.0, 0.3, step=0.05)
    min_conf = c2.slider("Confiance minimum", 0.1, 1.0, 0.6, step=0.05)

    if st.button("Extraire les règles", type="primary",
                 icon=":material/play_arrow:", width="stretch"):
        # construire la liste de transactions
        if mode.startswith("Colonne"):
            transactions = [set(str(v).split(",")) for v in df[col_panier].dropna()]
        else:
            cat = df.select_dtypes(exclude="number").columns
            transactions = [
                {f"{c}={row[c]}" for c in cat} for _, row in df.iterrows()
            ]

        regles = regles_association(transactions, min_sup, min_conf)
        st.metric("Nombre de règles trouvées", len(regles))

        if regles:
            tab = pd.DataFrame(
                [
                    {
                        "Si (antécédent)": ", ".join(sorted(r["antecedent"])),
                        "Alors (conséquent)": ", ".join(sorted(r["consequent"])),
                        "Support": r["support"],
                        "Confiance": r["confidence"],
                        "Lift": r["lift"],
                    }
                    for r in regles
                ]
            )
            st.dataframe(tab, width="stretch")
            telecharger(tab, "regles_association.csv")

            st.subheader("Top règles par confiance")
            top = tab.head(10).copy()
            top["règle"] = top["Si (antécédent)"] + " → " + top["Alors (conséquent)"]
            st.bar_chart(top.set_index("règle")["Confiance"])
        else:
            st.warning("Aucune règle : baisse le support ou la confiance minimum.",
                       icon=":material/warning:")

        st.info("**Comparaison WEKA** : *Associate → Apriori*. Régler "
                "`lowerBoundMinSupport` et `minMetric` (confiance) aux mêmes valeurs ; "
                "comparer les règles et leurs *lift*.", icon=":material/science:")

    with st.expander("Rappels — support, confiance, lift", icon=":material/menu_book:"):
        st.latex(r"\text{support}(X)=\frac{\#\{\text{transactions} \supseteq X\}}{N}")
        st.latex(r"\text{confiance}(X\to Y)=\frac{\text{support}(X\cup Y)}{\text{support}(X)}")
        st.latex(r"\text{lift}(X\to Y)=\frac{\text{confiance}(X\to Y)}{\text{support}(Y)}"
                 r"\quad(>1:\text{corrélation positive})")
