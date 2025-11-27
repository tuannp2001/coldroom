# Coldroom Project – Modélisation hybride physique / IA

Ce projet a pour objectif de modéliser la dynamique thermique d’une petite chambre froide
(air intérieur, produit) à partir de données expérimentales.  
Nous utilisons plusieurs approches :

- **Modèles linéaires physiques** pour estimer des paramètres physiques concentrés du modèle RC
- **Modèles d’IA purs** (réseaux neuronaux récurrents / résiduels)
- **Modèles hybrides PIML** (Physics-Informed Machine Learning : PINN, NNiSS, NNrSS) qui combinent équations physiques et apprentissage automatique.
- À terme, des techniques d’**incertitude** et **Conformal Prediction** pour obtenir des intervalles de prédiction fiables.

L’objectif final est de disposer d’un outil capable de prédire les températures (air, eau…) en free-run 
---

## Structure du projet

### `Data_coldroom/`
Ce dossier contient toutes les **données nécessaires pour l’apprentissage et la validation** des modèles :

- séries temporelles de températures (air, eau, etc.)
- éventuels signaux de commande (ON/OFF ventilateur)
- splits train / validation.

Ces données ne sont pas modifiées directement par les scripts d’entraînement : elles servent de source propre.

---

### `Linear_model/`
Ce dossier contient l'implémentation **modèle régression linéaire** utilisé pour :

- ajuster des modèles physiques simples et estimer les parramètres concentrés du modèle RC (bilan thermique --> modèles RC avec parramètres concentrés )

---

### `PIML_implementation/`
Implémentation des approches **Physics-Informed / Physics-Guided** :

- **PINN** (Physics-Informed Neural Network)
- **NNiSS** (Le modèle du state-space est implanté complètement par réseaux neuronaux)
- **NNRSS** (Le modèle du state-space est raffinné résiduellement par réseaux neuronaux)

On y trouve :

- le code d’**entraînement** des modèles (boucles de training, losses physiques + data-driven)
- la sauvegarde du modèle.

---

### `Post-processing/`
Scripts pour l’**analyse et la comparaison des résultats** :

- calcul d’indices d’erreur (RMSE, MAE, etc.)
- génération de figures comparant :
  - données mesurées vs. prédictions
  - modèles purement physiques vs. IA vs. hybrides
- **calcul des intervalles de prédiction**.  
  Les méthodes de **Conformal Prediction** seront ajoutées ici.

---

### `Pre-processing/`
Tout ce qui concerne le **traitement initial des données brutes** :

- chargement des données issues des fichiers sources (fichiers CSV.)
- premières **analyses exploratoires** et figures d’inspection (statistiques descriptives, corrélations, plots temporels).

Les scripts de ce dossier produisent typiquement les fichiers propres qui sont ensuite utilisés dans `Data_coldroom/` directement par les modèles.

---

### `Results/`
Ce dossier contient les **résultats d’expérience** pour chaque modèle :

- métriques d’évaluation (comme fichiers texte, JSON, CSV…)
- figures finales prêtes à être utilisées dans des rapports / articles
- comparatifs entre différentes configurations de modèles ou d’hyperparamètres.

---

### `Simulation_Code/`
Code pour **l’inférence et la simulation en free-run** :

- scripts qui chargent un modèle entraîné
- génération de prédictions **en roll-out** (prédiction auto-régressive sur de longues séquences)

C’est le dossier à utiliser si l’on veut tester un modèle sur de nouveaux scénarios sans le réentraîner.

---

### `Trained_model/`
Dossier dédié au **stockage des modèles entraînés** :
- poids sauvegardés (`.keras`)

Ces modèles sont ensuite utilisés par les scripts de `Simulation_Code/` et de `Post-processing/`.

---

## Prérequis (exemple à adapter)

- Python 3.x
- Bibliothèques principales :
  - `numpy`, `pandas`
  - `matplotlib`
  - `tensorflow` / `keras` (selon l’implémentation)
---

## Utilisation typique (pipeline)

1. **Pré-traitement** :  
   Lancer les scripts dans `Pre-processing/` pour préparer les données propres.

2. **Entraînement des modèles** :  
   Utiliser :
   - `Linear_model/` pour les modèles physiques simples
   - `PIML_implementation/` pour PINN, NNiSS, NNRSS, etc.

3. **Simulation / Free-run** :  
   Charger les modèles sauvegardés dans `Trained_model/` et lancer les scripts dans `Simulation_Code/` pour générer des prédictions en roll-out.

4. **Analyse des résultats** :  
   Utiliser les scripts de `Post-processing/` pour comparer les modèles et, à terme, calculer des **intervalles de prédiction** (Conformal Prediction).

---

## Remarques

Ce projet est en cours de développement.  
Certaines parties (notamment l’implémentation complète de la **Conformal Prediction**) seront ajoutées et documentées ultérieurement.
