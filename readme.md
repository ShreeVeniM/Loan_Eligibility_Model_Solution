PROJECT OVERVIEW
Introduction
This project focuses on predicting loan eligibility using Logistic Regression and Random Forest classifiers. The analysis includes data loading, preprocessing, model training, evaluation, and cross-validation.

Directory Structure
.git: Contains Git version control data.
.gitattributes: Git attributes file.
main.py: The main script to run the project.
src: Source code directory containing modules for various tasks.
Key Components
main.py
The central script that coordinates the following tasks:

Data Loading:

Loads loan eligibility data from src/dataset/credit.csv.
Data Preprocessing:

Cleans and preprocesses the loaded data.
Splits the data into training and testing sets.
Scales the data for better model performance.
Model Training and Evaluation:

Trains and evaluates Logistic Regression and Random Forest models.
Evaluates the models using accuracy and confusion matrix.
Prints evaluation results for both models.
Cross-Validation:

Performs cross-validation for both Logistic Regression and Random Forest models.
Prints cross-validation results for both models.
requirements.txt
Specifies the Python libraries required for the project:

pandas
numpy
matplotlib
seaborn
scikit-learn

Conclusion
This project provides a comprehensive analysis and prediction of loan eligibility using Logistic Regression and Random Forest classifiers. It includes detailed evaluation metrics and cross-validation results to assess model performance