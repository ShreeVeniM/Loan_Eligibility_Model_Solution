def print_evaluation_results(acc, conf_matrix):
    try:
        print(f"Accuracy: {acc}")
        print(f"Confusion Matrix:\n {conf_matrix}")
    except Exception as e:
        print(f"Error printing evaluation results: {e}")

def print_cross_val_results(scores):
    try:
        print(f"Cross-validation scores: {scores}")
        print(f"Mean Accuracy: {scores.mean()}")
        print(f"Standard Deviation: {scores.std()}")
    except Exception as e:
        print(f"Error printing cross-validation results: {e}")
