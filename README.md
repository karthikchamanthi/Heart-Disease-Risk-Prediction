Heart Disease Risk Prediction — Model Tuning & Ensemble Comparison
Machine learning project that predicts heart disease risk from patient symptoms and risk factors, comparing a baseline Decision Tree against hyperparameter-tuned versions (GridSearchCV, RandomizedSearchCV) and a Random Forest ensemble.

Dataset
The heart_disease_risk_dataset_earlymed.csv dataset is a synthetic, publicly available dataset from 
Kaggle (EarlyMed)
 with 70,000 patient records and 19 columns (18 predictors + 1 target):

16 binary indicators of symptoms and risk factors (chest pain, shortness of breath, fatigue, palpitations, dizziness, swelling, high blood pressure, high cholesterol, diabetes, smoking, obesity, sedentary lifestyle, family history, chronic stress, etc.)

Gender and Age

Heart_Risk — target variable (0 = low risk, 1 = high risk), perfectly balanced (35,000 / 35,000)

Project Workflow
Data loading & EDA — shape, info, describe, missing values, duplicates, and per-feature boxplots

Train/test split — 80/20 split with random_state=42

Baseline model — DecisionTreeClassifier(random_state=42) evaluated with 5-fold cross-validation on the training split and on the held-out test set

Hyperparameter tuning (both with the same 5-fold CV, scoring='accuracy'):

GridSearchCV — exhaustive search over max_depth, min_samples_split, min_samples_leaf (4 × 4 × 4 = 64 combinations)

RandomizedSearchCV — 10 random combinations from the same grid

Ensemble — RandomForestClassifier(random_state=42) with 5-fold CV and test-set evaluation

Comparison — final table with Model, CV Accuracy, Test Accuracy, and Best Params

Results
Model	CV Accuracy	Test Accuracy	Best Params
Decision Tree (baseline)	0.9820	0.9808	Not tuned (defaults)
Decision Tree (GridSearchCV)	0.9749	0.9734	{'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2}
Decision Tree (RandomizedSearchCV)	0.9749	0.9734	{'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 2}
Random Forest	0.9919	0.9921	Not tuned (defaults)
Key Findings
Random Forest achieved the highest observed accuracy among the tested configurations (99.2% test accuracy).

The baseline Decision Tree already performs strongly (98.1% test accuracy).

Neither hyperparameter search improved upon the baseline within the tested parameter space, which capped max_depth at 8 while the default tree grows as deep as the data allows.

RandomizedSearchCV matched GridSearchCV at a lower search cost (10 vs. 64 candidate combinations, i.e. 50 vs. 320 cross-validation fits).

Methodology Notes
All CV scores are computed on the training split with the same 5-fold splitter (KFold(n_splits=5, shuffle=True, random_state=42)), so they are directly comparable; test accuracy is the final held-out estimate.

No feature scaling is applied — all four models are tree-based and invariant to monotonic feature scaling, which also avoids leakage from fitting a scaler on test data.

Fixed seeds are used for the data split, CV folds, estimators, and randomized sampling to make runs repeatable with the same data and software versions.

Note: the results above were produced by running the notebook on the publicly downloaded 
Kaggle EarlyMed dataset
; the data is synthetic, so the models are for educational purposes only and are not suitable for clinical use.

How to Run
Download heart_disease_risk_dataset_earlymed.csv from the 
Kaggle dataset page

Open TUNE_AND_ENSEMBLE.ipynb in Jupyter, Colab, or VS Code

Update the CSV path in the second cell if needed (default assumes /content/ for Google Colab)

Run all cells top to bottom


data set:https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset


Requirements
Python 3.x

pandas, numpy, matplotlib, seaborn, scikit-learn

bash
