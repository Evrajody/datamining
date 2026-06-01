import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

# =========================
# 1. Création du dataset
# =========================

data = {
    "Age": [23, 17, 43, 68, 32, 20],
    "Type_voit": ["Familiale", "Sport", "Sport", "Familiale", "Camion", "Familiale"],
    "Risque": ["Élevé", "Élevé", "Élevé", "Faible", "Faible", "Élevé"],
}

df = pd.DataFrame(data)

print("Dataset :")
print(df)

# =========================
# 2. Encodage des données
# =========================
# Les arbres sklearn travaillent
# avec des valeurs numériques.

df_encoded = pd.get_dummies(df, columns=["Type_voit"])

print("\nDataset encodé :")
print(df_encoded)

# =========================
# 3. Variables X et y
# =========================

X = df_encoded.drop("Risque", axis=1)

# Conversion de la cible

y = df_encoded["Risque"].map({"Faible": 0, "Élevé": 1})


# =========================
# 4. Création du modèle
# =========================

model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)

# =========================
# 5. Entraînement
# =========================

model.fit(X, y)

# =========================
# 6. Affichage de l'arbre
# =========================

plt.figure(figsize=(12, 6))

plot_tree(model, feature_names=X.columns, class_names=["Faible", "Élevé"], filled=True)

plt.title("Arbre de décision - Indice de Gini")

plt.show()
