# Adult Census Income Prediction
Ce projet vise à prédire si un individu gagne plus de 50 000 $ par an en se basant sur des données socio-économiques issues du recensement de 1994.

Le dataset que nous avons choisi est "Adult Census Income" vous pouvez le retrouver sur ce lien (https://www.kaggle.com/datasets/uciml/adult-census-income/data)

**Objectif :** Prédire la variable cible income (classification binaire : <=50K ou >50K).
* **Type de problème :** Classification supervisée sur des données tabulaires mixtes.

Version de Python : 3.12.0

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
### Installation et environnement
Créer un environnement virtuel Python puis installer les dépendances donnée dans le requirements.txt
##### Instructions d’installation
``` bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Reproduire les résultats
##### Ordre exact d’exécution des notebooks/scripts
Pour éxecuter les notebooks il suffit de suivre les numéros que nous avons mis en amont de chaque numéro de fichier 
### Résumé EDA
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
