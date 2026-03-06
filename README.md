# Toxicity data classification using ML
# Project overview
This repository contains a Jupyter Notebook that builds a machine learning model to predict whether a chemical compound is Toxic or NonToxic based on a large set of molecular descriptors.
The goal is to show a workflow that has:
  -Data preprocessing 
  -Feature selection 
  -Model training 
  -Cross‑validation 
  -Model evaluation 
  -Decision threshold tuning to improve minority class recall
# Dataset
The dataset (toxicity dataset.csv) contains 171 rows and 1203 columns.
Target distribution:

NonToxic: 115 (67.3%)
Toxic: 56 (32.7%)

No missing values were detected.
# Methodology
1. Preprocessing
Constant features removed (VarianceThreshold(threshold=0).
Low‑variance features removed (threshold=0.00001).
Feature scaling with StandardScaler.
Highly correlated features removed (Pearson correlation > 0.9) to reduce multicollinearity.

3. Feature selection
A Random Forest model was used to calculate feature importance scores.
Features with importance values below 0.001 were removed.

Final Dataset:
Samples-171	
Features-50
This step reduces noise and keeps only the most predictive variables.

4.Model Training
The final model used is a Random Forest classifier.

Why Random Forest?
-Resistant to overfitting
-Works well with non-linear relationships
-Provides feature importance rankings

Model performance was performed using cross-validation.

5. Model evaluation
The following metrics were used:
Recall
Precision
ROC-AUC
F1 Score

| Class    | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| NonToxic | 0.72      | 0.75   | 0.73     |
| Toxic    | 0.40      | 0.36   | 0.38     |
Overall Accuracy: 0.63
ROC-AUC: 0.659

6.Results Interpretation
The model performs well in identifying Non-Toxic samples, but struggles more with detecting Toxic samples.This happens because:
-The classes are imbalanced
-The dataset is small (171 samples)
-The Toxic class has fewer example

Despite these limitations, the model achieves a moderate ROC-AUC score (~0.66), indicating it still captures useful predictive patterns.

7.Technologies used
Jupyter notebook
Python
Matplotlib
Seaborn
Pandas
NumPy
Scikit-learn
8. To run the project:
-Clone the repository:
git clone https://github.com/yourusername/toxicity-prediction.git
-Navigate into the project directory:
cd toxicity-prediction
-Install required dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn
-Run the notebook:
jupyter notebook
Project Structure
toxicity-prediction
│
├── data
│   └── dataset.csv
│
├── notebooks
│   └── toxicity_model.ipynb
│
├── README.md
└── requirements.txt
 9.Key Outcomes
This project demonstrates:
-Model evaluation using cross-validation
-Feature reduction in high-dimensional datasets
-Use of Random Forest for classification
-Interpretation of feature importance
-Communicating machine learning results effectively
