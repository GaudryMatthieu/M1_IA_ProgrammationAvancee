# Titre du projet + description courte
### Lien du dataset + ce que vous prédisez / analysez
Le dataset que nous avons choisi est "Adult Census Income" vous pouvez le retrouver sur ce lien (https://www.kaggle.com/datasets/uciml/adult-census-income/data)
``` python
import kagglehub
import pandas as pd
import os

# 1. Télécharger la dernière version du dataset
path = kagglehub.dataset_download("uciml/adult-census-income")

print("Chemin vers les fichiers du dataset :", path)

# 2. Lister les fichiers pour trouver le nom exact du CSV
files = os.listdir(path)
print("Fichiers disponibles :", files)

# 3. Charger le fichier CSV dans un DataFrame Pandas
file_path = os.path.join(path, "adult.csv")
df = pd.read_csv(file_path)

# 4. Affichage des premières lignes pour faire nos prmeier tests sur ce dataset
print(df.head())
```
### Type de problème (classification/régression/etc.)
### Installation et environnement
##### Version de Python
##### Instructions d’installation
### Reproduire les résultats
##### Ordre exact d’exécution des notebooks/scripts
### Résumé EDA (puces + figures clés)
### Résumé de modélisation
##### Modèles entraînés
##### Tableau des métriques
##### Modèle final choisi + justification
### Résumé du tuning d’hyperparamètres
##### Ce que vous avez tuné, pourquoi, et quels gains
### Section analyse d’erreurs
### Explication de la structure du projet
### Limites / pistes d’amélioration
### Références
