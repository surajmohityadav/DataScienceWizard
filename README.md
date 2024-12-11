
# Loan Default Prediction - Classification Model

## Project Overview

This project aims to develop a classification model to predict loan default behavior for a Non-Banking Financial Company (NBFC). The primary goal is to identify potential defaulters and non-defaulters from historical loan data, thereby improving the loan approval process and risk assessment. The dataset contains loan application details from the past two years, and the goal is to predict loan status (defaulter or non-defaulter) based on various customer attributes.

### Problem Statement:
We are tasked with predicting the loan default status (1: default, 0: non-default) based on attributes like customer demographics, loan details, and financial history.

## Dataset

The data is provided in two parts:
- **train_data.xlsx**: Historical loan data used for model training.
- **test_data.xlsx**: Recent loan data used for model evaluation.

### Columns:
- `customer_id`: Unique ID for each customer
- `transaction_date`: Date of the transaction
- `sub_grade`: Customer's sub-grade (based on geography, income, age)
- `term`: Loan tenure
- `home_ownership`: Home ownership status of the applicant
- `cibil_score`: CIBIL score of the applicant
- `total_no_of_acc`: Total number of bank accounts held by the applicant
- `annual_inc`: Annual income of the applicant
- `int_rate`: Interest rate charged
- `purpose`: Purpose of taking the loan
- `loan_amnt`: Total loan amount
- `application_type`: Type of application
- `installment`: Installment amount
- `verification_status`: Status of applicant verification
- `account_bal`: Account balance of the applicant
- `emp_length`: Years of employment experience
- `loan_status`: Loan status (1: default, 0: non-default)

---

## Approach

### 1. **Exploratory Data Analysis (EDA)**

- Performed EDA on the provided training data to understand the distribution of various features and their relationships with the target variable (`loan_status`).
- Visualized key insights through charts such as histograms, correlation heatmaps, and distribution plots.

### 2. **Data Preprocessing**

- Preprocessed the data by handling missing values, encoding categorical features, and scaling numerical features.
- Applied **One-Hot Encoding** to categorical features to make them suitable for model training.

### 3. **Modeling**

- Implemented **XGBoost** and **LightGBM** models for classification. Both models were chosen for their high efficiency and scalability for large datasets.
- Used a class-based approach to define model structure, including functions like:
  - `load`: Load data
  - `preprocess`: Data preprocessing
  - `train`: Model training
  - `test`: Testing and generating evaluation metrics
  - `predict`: Inference for new data

### 4. **Hyperparameter Tuning**

- Performed **GridSearchCV** to fine-tune the hyperparameters for both XGBoost and LightGBM models.
  - **XGBoost** hyperparameters: `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`
  - **LightGBM** hyperparameters: `learning_rate`, `n_estimators`, `max_depth`, `num_leaves`, `subsample`

### 5. **Model Evaluation**

- Evaluated the models based on several metrics: accuracy, precision, recall, F1-score, and confusion matrix.
- Selected the best-performing model based on the highest **accuracy** and **recall for defaulters**.

---

## Results

### **XGBoost Model (Before Hyperparameter Tuning)**:
- **Accuracy**: 68.08%
- **Classification Report**:
  - Precision for non-defaulters (0): 0.64, for defaulters (1): 0.69
  - Recall for non-defaulters (0): 0.27, for defaulters (1): 0.91
  - F1-score for non-defaulters (0): 0.38, for defaulters (1): 0.79
- **Confusion Matrix**:
  - Non-defaulters (0): 818 True Negatives, 2237 False Positives
  - Defaulters (1): 462 False Negatives, 4938 True Positives

### **XGBoost Model (After Hyperparameter Tuning)**:
- **Accuracy**: 71.27%
- **Classification Report**:
  - Precision for non-defaulters (0): 0.67, for defaulters (1): 0.73
  - Recall for non-defaulters (0): 0.41, for defaulters (1): 0.89
  - F1-score for non-defaulters (0): 0.51, for defaulters (1): 0.80
- **Confusion Matrix**:
  - Non-defaulters (0): 1246 True Negatives, 1809 False Positives
  - Defaulters (1): 620 False Negatives, 4780 True Positives

### **LightGBM Model (Before Hyperparameter Tuning)**:
- **Accuracy**: 68.01%
- **Classification Report**:
  - Precision for non-defaulters (0): 0.63, for defaulters (1): 0.69
  - Recall for non-defaulters (0): 0.28, for defaulters (1): 0.91
  - F1-score for non-defaulters (0): 0.39, for defaulters (1): 0.79
- **Confusion Matrix**:
  - Non-defaulters (0): 838 True Negatives, 2217 False Positives
  - Defaulters (1): 469 False Negatives, 4934 True Positives

### **LightGBM Model (After Hyperparameter Tuning)**:
- **Accuracy**: 69.77%
- **Classification Report**:
  - Precision for non-defaulters (0): 0.69, for defaulters (1): 0.70
  - Recall for non-defaulters (0): 0.30, for defaulters (1): 0.92
  - F1-score for non-defaulters (0): 0.42, for defaulters (1): 0.80
- **Confusion Matrix**:
  - Non-defaulters (0): 912 True Negatives, 2143 False Positives
  - Defaulters (1): 413 False Negatives, 4987 True Positives

---

## Model Selection

Based on the evaluation metrics, the **XGBoost model** was selected as the final model due to its higher accuracy and better performance in identifying defaulters. Specifically, it had a higher **accuracy (71.27%)**, **recall for defaulters**, and overall better classification results compared to LightGBM.

**Reasons for choosing XGBoost**:
- Higher accuracy and recall for defaulters (class 1).
- Efficient performance even with large datasets.
- Robust model performance after hyperparameter tuning.

---

## Conclusion

The XGBoost model, after hyperparameter tuning, is the best choice for loan default prediction based on its accuracy and ability to classify defaulters correctly. This model can now be deployed for real-time prediction of loan default risks, assisting in more informed loan approval decisions for the NBFC.

---

## Files Included

1. **eda.ipynb** - Exploratory Data Analysis
2. **model.py** - Model implementation (XGBoost and LightGBM)
3. **model_selection.ipynb** - Model selection and evaluation
4. **xgboost_model.pkl** - Trained XGBoost model
5. **lightgbm_model.pkl** - Trained LightGBM model

