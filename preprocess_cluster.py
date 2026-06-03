import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# 1. Chargement des données
# Remplacez par le chemin de votre fichier si nécessaire
url = "/home/evrajodygildas/dev-laboratory/me.projects/master_2/datamining/datasets/CC GENERAL.csv"
df = pd.read_csv(url)

print("--- Données initiales ---")
print(df.head())
# Affiche le nombre exact de cases vides pour chaque colonne
print(df.isnull().sum())
# 1. Calcul des médianes
median_credit = df["CREDIT_LIMIT"].median()
median_payments = df["MINIMUM_PAYMENTS"].median()

# 2. Remplacement des valeurs manquantes (NaN) par les médianes trouvées
df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(median_credit)
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(median_payments)

# 3. Vérification du résultat
print("--- Vérification des valeurs manquantes après correction ---")
print(df[["CREDIT_LIMIT", "MINIMUM_PAYMENTS"]].isnull().sum())

# Calcule la corrélation absolue entre toutes les variables numériques
corr_matrix = df.select_dtypes(include=["number"]).corr().abs()
print(corr_matrix)

