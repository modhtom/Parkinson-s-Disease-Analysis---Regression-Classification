# Parkinson's Disease Analysis - Regression & Classification

Dual-mode machine learning system for Parkinson's disease analysis:
1. Regression: Predict UPDRS scores (Unified Parkinson's Disease Rating Scale)
2. Classification: Diagnose Parkinson's disease presence

## Key Features 

- Dual functionality (Regression & Classification)
- Comprehensive feature engineering:
  - Time conversion (HH:MM to minutes)
  - JSON column expansion
  - Custom encoding schemes
- Advanced model ensembles:
  - Voting Classifier
  - Stacking Classifier
- Detailed performance metrics:
  - MSE/RMSE/R² for regression
  - Accuracy/F1-scores for classification
  - Training/testing time analysis

## Usage 

Run analysis pipelines:
```bash
# Regression analysis
python app_regression.py

# Classification analysis
python app_classification.py

# Test models
python test.py --type [reg/cls] --path [test_data_path]
```
## DataSet
[DOWNLOAD](https://drive.google.com/file/d/1rhd5LI_iY4w7mocRVHaJPBGxF61sylb7/view?usp=drive_link)

## Model Performance 

### Regression Results (UPDRS Prediction)

- Random Forest       |Test MSE 0.8249
- Linear Regression   |Test MSE 0.7620
- Lasso Regression    |Test MSE 0.7707
- Ridge Regression    |Test MSE 0.7620

### Classification Results (Diagnosis)
- Stacking Classifier  93.58%
- Random Forest        91.36%
- Voting Classifier    89.38%
- SVM                  82.22%
