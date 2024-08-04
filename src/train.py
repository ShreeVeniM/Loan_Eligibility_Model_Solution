from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def scale_data(xtrain, xtest):
    from sklearn.preprocessing import MinMaxScaler
    try:
        scaler = MinMaxScaler()
        xtrain_scaled = scaler.fit_transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        return xtrain_scaled, xtest_scaled
    except Exception as e:
        print(f"Error scaling data: {e}")
        return None, None

def train_logistic_regression(xtrain_scaled, ytrain):
    try:
        model = LogisticRegression()
        model.fit(xtrain_scaled, ytrain)
        return model
    except Exception as e:
        print(f"Error training Logistic Regression model: {e}")
        return None

def train_random_forest(xtrain, ytrain):
    try:
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features='sqrt')
        model.fit(xtrain, ytrain)
        return model
    except Exception as e:
        print(f"Error training Random Forest model: {e}")
        return None
