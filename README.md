# Streamlit Auto Machine Learning APP

## Description
Ce projet est une application interactive développée avec Streamlit permettant de réaliser automatiquement des tâches de Machine Learning à partir d’un fichier CSV.
L’application permet aux utilisateurs de charger leurs propres données et d’entraîner plusieurs modèles de classification ou de régression afin de comparer leurs performances et effectuer des prédictions.

## Objectifs
- Permettre le téléchargement et l’exploration de jeux de données CSV
- Automatiser l’entraînement de modèles de Machine Learning
- Comparer plusieurs algorithmes de classification et de régression
- Fournir une interface de prédiction interactive
- Faciliter la prise de décision basée sur les performances des modèles

## Fonctionnalités

1. Importation des données
- Chargement dynamique de fichiers CSV
- Visualisation des premières lignes du dataset

2. Module de classification
- Sélection de la variable cible
- Entraînement de plusieurs modèles :
  - Régression logistique
  - K plus proches voisins
  - Support Vector Machine
  - Forêt aléatoire
- Affichage des métriques :
  - Accuracy
  - Précision
  - Recall
  - F1-score

2. Module de régression
- Sélection de la variable cible
- Entraînement de plusieurs modèles :
  - Régression linéaire
  - KNN Regressor
  - ElasticNet
  - Forêt aléatoire
- Évaluation des modèles :
  - MAE
  - MSE
  - R²

3. Interface de prédiction
- Saisie des nouvelles données par l’utilisateur
- Sélection du modèle entraîné
- Génération de prédictions en temps réel

4. Sauvegarde des modèles
- Sauvegarde automatique des modèles entraînés au format `.pkl`
- Chargement des modèles pour prédiction

## Jeux de données
L’application ne dépend pas d’un dataset fixe.
L’utilisateur peut importer n’importe quel fichier CSV contenant des variables numériques adaptées à des tâches de classification ou de régression.

## Technologies utilisées
- Python
- Streamlit
- Pandas
- Scikit-learn
- Pickle

## Exécution du projet
1. Installer les dépendances :
pip install streamlit pandas scikit-learn
2. Lancer l’application :
streamlit run app.py

## Auteur
Maniche Darlin Kamgang
Développeur Web, Mobile et Intelligence Artificielle