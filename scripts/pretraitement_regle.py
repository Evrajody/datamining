import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# 1. Chargement des données
# Remplacez par le chemin de votre fichier si nécessaire
url = "C:/Users/PC/Desktop/IG2/Data maning/projet/Wholesale customers data.csv"
df = pd.read_csv(url)

print("--- Données initiales ---")
print(df.head())
# Affiche le nombre exact de cases vides pour chaque colonne
print(df.isnull().sum())
# Écrivez ici les noms exacts des colonnes hors sujets à supprimer
colonnes_hors_sujets = ["Channel", "Region"]

# Suppression des colonnes
df_clean = df.drop(columns=colonnes_hors_sujets)

# Vérification : affiche les colonnes restantes
print("Colonnes restantes :", df_clean.columns.tolist())

# 1. Redéfinir les colonnes numériques sur votre jeu de données nettoyé
num_cols = df_clean.select_dtypes(include=["number"]).columns

# 1. Définir les bornes et les noms
bornes = [-np.inf, 500, 1000, 5000, 10000, np.inf]
noms_tranches = ["Très faible", "Faible", "Moyen", "Fort", "Très fort"]

# 2. Identifier vos colonnes de dépenses (ex: toutes sauf Channel et Region s'ils y sont encore)
# Ou utilisez directement votre liste 'num_cols' si elle est déjà définie plus haut
colonnes_depenses = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]

# 3. Appliquer le découpage sur chaque colonne avec une boucle
for col in colonnes_depenses:
    if col in df_clean.columns:  # Sécurité pour vérifier que la colonne existe
        df_clean[col] = pd.cut(
            df_clean[col], bins=bornes, labels=noms_tranches, right=False
        )
# 5. Enregistrer le fichier final propre
df_clean.to_csv("wholesale_nettoye2.csv", index=False)

print("Fichier 'wholesale_nettoye2.csv' enregistré avec succès et prêt pour WEKA !")
