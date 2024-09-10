# Credit Card Fraud Detection Project

## Purpose
The objective of this project is to develop a machine learning model capable of accurately predicting fraudulent credit card transactions. This will help credit card companies prevent unauthorized charges and protect their customers from financial losses.

## Overview
Credit card fraud poses a significant threat to financial transactions. This project focuses on building a classification model to determine whether a transaction is fraudulent. The dataset comprises transactions made by European cardholders in September 2013, totaling 284,807 transactions, of which only 492 are fraudulent. This imbalance necessitates careful handling to ensure the model’s effectiveness.

## Problem Statement
The objective of this project is to develop a robust classification model capable of accurately identifying fraudulent credit card transactions. By achieving this, credit card companies can prevent unauthorized charges and protect their customers from financial losses.

## Setup Instructions
To run the project, please follow these steps:

1. **Download Dataset**: Obtain the ‘creditcard.csv’ dataset.

2. **Install Dependencies**: Ensure the following Python libraries are installed:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - imbalanced-learn

3. **Run the Script**: Execute the provided Python script PredictionOfCreditCardFraud.ipynb to train and evaluate the model.

## Data Sources
The dataset `creditcard.csv` was used for this project. It contains the following attributes:
- **Time**: Time elapsed in seconds between each transaction and the first transaction.
- **V1 - V28**: Principal components obtained with PCA (anonymized features).
- **Amount**: Transaction amount.
- **Class**: Indicates whether the transaction is fraudulent (1) or not (0).

## Code and Models
### Code Explanation
The Python code performs the following tasks:
1. **Data loading and preprocessing**: Loads the dataset, handles missing values, converts data types, and performs oversampling to address class imbalance.
2. **Feature Engineering and Selection**: Adds new features and selects relevant features using PCA and ANOVA F-test.
3. **Model Selection and Training**: Utilizes a Random Forest Classifier for training the model and evaluates its performance using cross-validation.
4. **Hyperparameter Tuning**: Uses GridSearchCV to find the best hyperparameters for the model.
5. **Model Evaluation**: Evaluates the model's performance on test data, including accuracy, confusion matrix, and classification report.
6. **Additional Visualizations**: Includes visualizations such as correlation matrix, scatter plot, and transaction volume over time.

### Instructions for Running the Code
1. Install necessary libraries:
    ```bash
    pip install pandas numpy matplotlib scikit-learn imbalanced-learn joblib seaborn
    ```
2. Clone the repository and navigate to the project directory.
3. Ensure the `creditcard.csv` dataset is placed in the same directory.
4. Run the Python script (`PredictionOfCreditCardFraud.ipynb`).
6. The output will include cross-validation scores, model evaluation metrics, and additional visualizations.

### Reproducing Results
You can reproduce the results by following the instructions provided above and referring to the specific output metrics mentioned in the code explanations.

## Deployment in AWS SageMaker
For deployment in AWS SageMaker, you can use the trained model (credit_card_fraud_detection_model.pkl). Follow the AWS SageMaker documentation for deploying models and serving predictions.

---

## Project Structure
- `PredictionOfCreditCardFraud.ipynb`: Main Python script for model building and evaluation.
- `creditcard.csv`: Dataset containing credit card transactions.
- `Prediction_Of_Credit_Card_Fraud.pkl`: Saved trained model.
