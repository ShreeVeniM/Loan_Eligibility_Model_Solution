import logging
from src.data_loader import load_data
from src.data_preprocess import preprocess_data, split_data
from src.train import scale_data, train_logistic_regression, train_random_forest
from src.evaluate import evaluate_model, cross_validate_model
from src.utils import print_evaluation_results, print_cross_val_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info('Starting main function')
    try:
        file_path = 'src/dataset/credit.csv'
        
        # Load and preprocess data
        logging.info('Loading data from credit.csv')
        df = load_data(file_path)
        if df is None:
            logging.error('Failed to load data')
            return

        logging.info('Preprocessing data')
        df = preprocess_data(df)
        if df is None:
            logging.error('Failed to preprocess data')
            return
        
        # Split data into training and testing sets
        logging.info('Splitting data into training and testing sets')
        xtrain, xtest, ytrain, ytest = split_data(df)
        if xtrain is None or xtest is None or ytrain is None or ytest is None:
            logging.error('Failed to split data')
            return
        
        # Scale the data
        logging.info('Scaling data')
        xtrain_scaled, xtest_scaled = scale_data(xtrain, xtest)
        if xtrain_scaled is None or xtest_scaled is None:
            logging.error('Failed to scale data')
            return
        
        # Train and evaluate Logistic Regression model
        logging.info('Training Logistic Regression model')
        lr_model = train_logistic_regression(xtrain_scaled, ytrain)
        if lr_model is not None:
            logging.info('Evaluating Logistic Regression model')
            lr_acc, lr_conf_matrix = evaluate_model(lr_model, xtest_scaled, ytest)
            if lr_acc is not None and lr_conf_matrix is not None:
                logging.info('Printing evaluation results for Logistic Regression model')
                print_evaluation_results(lr_acc, lr_conf_matrix)
        
        # Cross-validation for Logistic Regression
        logging.info('Performing cross-validation for Logistic Regression model')
        lr_cv_scores = cross_validate_model(lr_model, xtrain_scaled, ytrain)
        if lr_cv_scores is not None:
            logging.info('Printing cross-validation results for Logistic Regression model')
            print_cross_val_results(lr_cv_scores)
        
        # Train and evaluate Random Forest model
        logging.info('Training Random Forest model')
        rf_model = train_random_forest(xtrain, ytrain)
        if rf_model is not None:
            logging.info('Evaluating Random Forest model')
            rf_acc, rf_conf_matrix = evaluate_model(rf_model, xtest, ytest)
            if rf_acc is not None and rf_conf_matrix is not None:
                logging.info('Printing evaluation results for Random Forest model')
                print_evaluation_results(rf_acc, rf_conf_matrix)
        
        # Cross-validation for Random Forest
        logging.info('Performing cross-validation for Random Forest model')
        rf_cv_scores = cross_validate_model(rf_model, xtrain_scaled, ytrain)
        if rf_cv_scores is not None:
            logging.info('Printing cross-validation results for Random Forest model')
            print_cross_val_results(rf_cv_scores)
            
    except Exception as e:
        logging.error(f'Error occurred: {e}')

if __name__ == "__main__":
    main()
