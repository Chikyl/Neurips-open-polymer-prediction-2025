#NeurIPS 2025 – Open Polymer Prediction

Machine Learning for Polymer Property Prediction Using Molecular Representations

##Overview

This project explores machine learning techniques for predicting polymer properties from molecular structures represented as SMILES (Simplified Molecular Input Line Entry System).

Developed for the NeurIPS 2025 Open Polymer Prediction Challenge, the approach combines cheminformatics-based feature engineering with gradient-boosting regression to predict five important polymer properties:

Tg: Glass Transition Temperature

FFV: Fractional Free Volume

Tc: Thermal Conductivity

Density: Polymer Density

Rg: Radius of Gyration

##Methodology

The solution implements an end-to-end machine learning pipeline consisting of:

Data Preparation: Loading competition datasets and incorporating supplementary polymer data.

Molecular Standardization: Using RDKit to convert SMILES representations into canonical SMILES for consistent molecular identification.

Feature Engineering: Extracting 14 molecular descriptors, including molecular weight, LogP, TPSA, hydrogen bond donors and acceptors, and functional group counts.

Morgan Fingerprints: Generating 1024-bit molecular fingerprints with a radius of 2 to capture molecular substructures.

Hyperparameter Optimization: Applying Optuna to identify target-specific XGBoost hyperparameters.

Model Training: Training five independent XGBoost regression models, one for each polymer property.

Evaluation: Using five-fold GroupKFold cross-validation with canonical SMILES as grouping identifiers and Mean Absolute Error (MAE) as the optimization metric.

Prediction: Generating predictions for the test dataset and preparing the final competition submission.





