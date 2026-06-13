import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/PC/Desktop/IG2/Data maning/projet/CC GENERAL.csv")

# 1. Supprimer colonnes inutiles (ex: ID client)
df = df.drop(columns=["CUST_ID"], errors="ignore")

# 2. Imputer les valeurs manquantes
df = df.fillna(df.median(numeric_only=True))

# 3. Plafonner les outliers (Winsorization au 90ème percentile)
# On identifie les colonnes numériques sur lesquelles appliquer le plafonnement
num_cols = df.select_dtypes("number").columns

for col in num_cols:
    # On calcule le seuil du 90ème percentile (on garde les 90% les plus bas)
    seuil_haut = df[col].quantile(0.90)

    # Si une valeur dépasse ce seuil, on la remplace (la plafonne) par la valeur du seuil
    df[col] = df[col].clip(upper=seuil_haut)

# Note : Plus besoin de reset_index car aucune ligne n'a été supprimée !
# 4. Standardiser
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df.select_dtypes("number")),
    columns=df.select_dtypes("number").columns,
)

df_scaled.to_csv("cc_general.csv", index=False)
print(f"Données prêtes : {df_scaled.shape[0]} lignes, {df_scaled.shape[1]} colonnes")
