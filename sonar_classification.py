# ============================================================
# SONAR - Classification Mines vs Rochers
# Dataset : Gorman & Sejnowski (1988)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

# 60 colonnes de features + 1 colonne label (R ou M)
colonnes = [f"freq_{i}" for i in range(1, 61)] + ["label"]

df = pd.read_csv("sonar_all-data.csv", header=None, names=colonnes)

print("=== Aperçu des données ===")
print(df.head())
print(f"\nDimensions : {df.shape}")
print(f"Classes    : {df['label'].value_counts().to_dict()}")

# ============================================================
# 2. PRÉPARATION DES DONNÉES
# ============================================================

X = df.drop("label", axis=1).values          # Features (60 colonnes)
y = (df["label"] == "M").astype(int).values  # 1 = Mine, 0 = Rocher

# Séparation entraînement / test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nEntraînement : {X_train.shape[0]} échantillons")
print(f"Test         : {X_test.shape[0]} échantillons")

# ============================================================
# 3. MODÈLES
# ============================================================

modeles = {
    "Réseau de neurones (MLP)": MLPClassifier(
        hidden_layer_sizes=(24,), max_iter=1000, random_state=42
    ),
    "K plus proches voisins": KNeighborsClassifier(n_neighbors=5),
}

resultats = {}

for nom, modele in modeles.items():
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    resultats[nom] = acc

    print(f"\n{'='*50}")
    print(f"  {nom}")
    print(f"{'='*50}")
    print(f"Précision : {acc*100:.1f}%")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=["Rocher", "Mine"]))

# ============================================================
# 4. VISUALISATIONS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SONAR — Classification Mines vs Rochers", fontsize=14, fontweight="bold")

# --- Matrice de confusion (meilleur modèle) ---
meilleur_nom = max(resultats, key=resultats.get)
meilleur_modele = modeles[meilleur_nom]
y_pred_best = meilleur_modele.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rocher", "Mine"],
            yticklabels=["Rocher", "Mine"],
            ax=axes[0])
axes[0].set_title(f"Matrice de confusion\n({meilleur_nom})")
axes[0].set_xlabel("Prédit")
axes[0].set_ylabel("Réel")

# --- Comparaison des modèles ---
noms  = list(resultats.keys())
accs  = [v * 100 for v in resultats.values()]
bars  = axes[1].bar(noms, accs, color=["#2196F3", "#4CAF50"])
axes[1].set_ylim(0, 100)
axes[1].set_ylabel("Précision (%)")
axes[1].set_title("Comparaison des modèles")
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1, f"{acc:.1f}%",
                 ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("sonar_resultats.png", dpi=150)
plt.show()
print("\n✅ Graphique sauvegardé : sonar_resultats.png")