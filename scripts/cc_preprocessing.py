"""
PROJET — SEGMENTATION DE TITULAIRES DE CARTES DE CRÉDIT
========================================================
Chaîne de PRÉTRAITEMENT optimale du dataset « CC GENERAL.csv » (8 950 clients,
17 variables comportementales sur 6 mois), jusqu'au LANCEMENT de l'analyse
(choix du nombre de clusters k + premier K-Means).

Pourquoi ces étapes, et dans cet ordre ?
  0. Charger + retirer l'identifiant  -> CUST_ID n'est pas une mesure, il fausse les distances.
  1. Diagnostic qualité               -> on ne corrige que ce qu'on a mesuré (NaN, doublons).
  2. Imputation par la MÉDIANE        -> robuste aux distributions très asymétriques.
  3. Transformation LOG (log1p)       -> écrase les longues queues / quasi-symétrise. (INCOMPRIS)
  4. Winsorisation légère (1%/99%)    -> borne les valeurs extrêmes sans supprimer de clients.
  5. STANDARDISATION (z-score)        -> INDISPENSABLE : K-Means et l'ACP sont basés sur la
                                         distance euclidienne ; sans elle, BALANCE (≈ milliers)
                                         écraserait les fréquences (∈ [0,1]).
  6. Diagnostic de CORRÉLATION        -> repérer la redondance (PURCHASES vs ONEOFF/INSTALLMENTS).
  7. ACP                              -> visualiser en 2D et mesurer la variance expliquée.
  8. Choix de k : COUDE + SILHOUETTE  -> deux critères convergents, pas un seul. (INCOMPRIS)
  9. Premier K-Means + profils        -> on « lance » l'analyse et on interprète les centres.

Exécution :  uv run python scripts/cc_preprocessing.py
Les figures sont enregistrées dans rapport/figures/ pour le rapport LaTeX.
"""

import os

import matplotlib

matplotlib.use("Agg")  # backend sans écran : on enregistre les figures en PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ICI = os.path.dirname(__file__)
FIG = os.path.join(ICI, "..", "rapport", "figures")
DATA = os.path.join(ICI, "..", "datasets", "CC GENERAL.csv")
os.makedirs(FIG, exist_ok=True)
RNG = 42  # graine fixe -> résultats reproductibles


# ===========================================================================
# 0. CHARGEMENT + RETRAIT DE L'IDENTIFIANT
# ===========================================================================
# CUST_ID est une étiquette (C10001, ...), pas une grandeur mesurable.
# La garder reviendrait à injecter du bruit dans le calcul des distances.
df = pd.read_csv(DATA)
print(f"Dimensions brutes : {df.shape[0]} clients × {df.shape[1]} colonnes")
df = df.drop(columns=["CUST_ID"])


# ===========================================================================
# 1. DIAGNOSTIC DE QUALITÉ
# ===========================================================================
# On NE corrige que ce qu'on a constaté. Deux choses à vérifier :
#   - les valeurs manquantes (NaN)
#   - les doublons éventuels (clients en double)
print("\n--- Valeurs manquantes par colonne ---")
manquants = df.isnull().sum()
print(manquants[manquants > 0])
print(f"\nLignes dupliquées : {df.duplicated().sum()}")


# ===========================================================================
# 2. IMPUTATION PAR LA MÉDIANE
# ===========================================================================
# MINIMUM_PAYMENTS (~313 NaN) et CREDIT_LIMIT (1 NaN).
# On choisit la MÉDIANE plutôt que la moyenne : ces variables sont très
# asymétriques (quelques très gros payeurs), or la moyenne y est tirée vers
# le haut par les valeurs extrêmes ; la médiane reste représentative.
for col in ["MINIMUM_PAYMENTS", "CREDIT_LIMIT"]:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
print("\nNaN restants après imputation :", int(df.isnull().sum().sum()))


# ===========================================================================
# 3. TRANSFORMATION LOGARITHMIQUE (log1p) (INCOMPRIS)
# ===========================================================================
# La majorité des variables monétaires/comptage sont très asymétriques
# (skewness ≫ 1) : beaucoup de 0 et de petites valeurs, quelques très grandes.
# log1p(x) = log(1 + x) :
#   - définie en 0 (pas de log(0)),
#   - compresse les longues queues -> distributions plus symétriques,
#   - rend les distances euclidiennes plus « justes » (un écart de 50 € à bas
#     niveau pèse autant qu'un écart de 5 000 € à haut niveau).
# On NE transforme PAS les variables déjà bornées dans [0, 1] (les *_FREQUENCY)
# ni TENURE (faible amplitude) : la log les déformerait sans bénéfice.
freq_cols = [c for c in df.columns if "FREQUENCY" in c]
exclues_log = set(freq_cols + ["TENURE"])
a_logger = [c for c in df.columns if c not in exclues_log]

skew_avant = df[a_logger].skew()
df[a_logger] = np.log1p(df[a_logger])
skew_apres = df[a_logger].skew()
print("\n--- Asymétrie (skewness) avant -> après log1p ---")
print(pd.DataFrame({"avant": skew_avant, "après": skew_apres}).round(2))


# ===========================================================================
# 4. WINSORISATION LÉGÈRE (1 % / 99 %) (INCOMPRIS)
# ===========================================================================
# Après la log il reste des extrêmes. Plutôt que de SUPPRIMER des clients
# (on perdrait de l'information métier), on BORNE chaque variable à ses
# percentiles 1 % et 99 %. K-Means étant sensible aux outliers (la moyenne
# définit les centres), cela stabilise les centres sans réduire l'effectif.
bornes = df.quantile([0.01, 0.99])
df = df.clip(lower=bornes.loc[0.01], upper=bornes.loc[0.99], axis=1)


# ===========================================================================
# 5. STANDARDISATION (Z-SCORE)  ->  X* = (X - moyenne) / écart-type
# ===========================================================================
# ÉTAPE CRITIQUE. K-Means et l'ACP reposent sur la distance euclidienne.
# Sans standardisation, une variable à grande amplitude (BALANCE, PAYMENTS…)
# domine totalement le calcul, et les fréquences ∈ [0,1] deviennent
# négligeables. Le z-score met toutes les variables sur un pied d'égalité
# (moyenne 0, écart-type 1).
scaler = StandardScaler()
X = scaler.fit_transform(df)
X = pd.DataFrame(X, columns=df.columns)
print(
    f"\nDonnées standardisées : moyenne≈{X.values.mean():.2e}, "
    f"écart-type≈{X.values.std():.2f}"
)


# ===========================================================================
# 6. DIAGNOSTIC DE CORRÉLATION (redondance)
# ===========================================================================
# But : repérer les variables redondantes (ex. PURCHASES ≈ ONEOFF + INSTALLMENTS).
# On les CONSERVE ici (l'ACP gérera la redondance), mais on documente le constat.
corr = X.corr().abs()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr.columns, fontsize=7)
ax.set_title("Matrice de corrélation absolue (après prétraitement)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cc_correlation.png"), dpi=130)
plt.close(fig)

paires = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)
print("\n--- Paires les plus corrélées (|r| > 0.8) ---")
print(paires[paires > 0.8].round(2))


# ===========================================================================
# 7. ACP — VARIANCE EXPLIQUÉE + PROJECTION 2D
# ===========================================================================
# L'ACP sert ici à (a) mesurer combien d'axes suffisent pour résumer
# l'information, et (b) visualiser les clusters en 2D.
pca = PCA(random_state=RNG).fit(X)
cumul = np.cumsum(pca.explained_variance_ratio_)
n_90 = int(np.argmax(cumul >= 0.90) + 1)
print(f"\nACP : {n_90} composantes expliquent ≥ 90 % de la variance.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cumul) + 1), cumul, "o-")
ax.axhline(0.90, color="red", ls="--", label="seuil 90 %")
ax.axvline(n_90, color="grey", ls=":")
ax.set_xlabel("Nombre de composantes principales")
ax.set_ylabel("Variance cumulée expliquée")
ax.set_title("ACP — variance expliquée cumulée")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cc_pca_variance.png"), dpi=130)
plt.close(fig)

X_2d = PCA(n_components=2, random_state=RNG).fit_transform(X)


# ===========================================================================
# 8. CHOIX DE k — MÉTHODE DU COUDE + SCORE SILHOUETTE (INCOMPRI)
# ===========================================================================
# On ne devine pas k : on le justifie avec DEUX critères.
#   - Coude (inertie intra-cluster) : on cherche le « pli » de la courbe.
#   - Silhouette (∈ [-1,1]) : cohésion vs séparation ; plus haut = mieux.
ks = range(2, 11)
inerties, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG)
    labels = km.fit_predict(X)
    inerties.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(list(ks), inerties, "o-")
a1.set_title("Méthode du coude (inertie)")
a1.set_xlabel("k")
a1.set_ylabel("Inertie intra-cluster")
a2.plot(list(ks), silhouettes, "o-", color="green")
a2.set_title("Score de silhouette")
a2.set_xlabel("k")
a2.set_ylabel("Silhouette moyenne")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cc_choix_k.png"), dpi=130)
plt.close(fig)

k_opt = list(ks)[int(np.argmax(silhouettes))]
print(
    f"\nMeilleur k selon la silhouette : {k_opt} (silhouette = {max(silhouettes):.3f})"
)


# ===========================================================================
# 9. LANCEMENT DE L'ANALYSE — K-MEANS FINAL + PROFILS
# ===========================================================================
km = KMeans(n_clusters=3, n_init=10, random_state=0)
df["CLUSTER"] = km.fit_predict(X)

# Projection ACP colorée par cluster
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=df["CLUSTER"], cmap="tab10", s=8, alpha=0.6)
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_title(f"Segmentation K-Means (k={k_opt}) projetée par ACP")
fig.colorbar(sc, ax=ax, label="cluster")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cc_clusters_acp.png"), dpi=130)
plt.close(fig)

# Profils : moyenne des variables ORIGINALES (lisibles) par cluster.
# On recharge le brut imputé pour interpréter en unités réelles (€, %).
brut = pd.read_csv(DATA).drop(columns=["CUST_ID"])
for col in ["MINIMUM_PAYMENTS", "CREDIT_LIMIT"]:
    brut[col] = brut[col].fillna(brut[col].median())
brut["CLUSTER"] = df["CLUSTER"]
profils = brut.groupby("CLUSTER").mean().round(1)
print(f"\n--- Profils moyens des {k_opt} segments (unités réelles) ---")
print(profils.T)
print("\nEffectifs par cluster :")
print(df["CLUSTER"].value_counts().sort_index())

# Sauvegarde des données prêtes pour la suite (interprétation, app Streamlit…)
sortie = os.path.join(ICI, "..", "datasets", "cc_clusters.csv")
brut.to_csv(sortie, index=False)
print(f"\nDonnées segmentées enregistrées -> {sortie}")
print("Figures enregistrées dans rapport/figures/ (cc_*.png)")
