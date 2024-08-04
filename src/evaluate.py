from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold

def evaluate_model(model, xtest_scaled, ytest):
    try:
        ypred = model.predict(xtest_scaled)
        acc = accuracy_score(ypred, ytest)
        conf_matrix = confusion_matrix(ytest, ypred)
        return acc, conf_matrix
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None, None

def cross_validate_model(model, xtrain, ytrain):
    try:
        kfold = KFold(n_splits=5)
        scores = cross_val_score(model, xtrain, ytrain, cv=kfold)
        return scores
    except Exception as e:
        print(f"Error in cross-validation: {e}")
        return None
