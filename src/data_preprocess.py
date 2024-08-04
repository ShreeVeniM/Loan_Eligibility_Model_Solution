import pandas as pd

def preprocess_data(df):
    try:
        df['Gender'].fillna('Male', inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        df.drop('Loan_ID', axis=1, inplace=True)
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'], dtype=int)
        return df
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def split_data(df):
    from sklearn.model_selection import train_test_split
    try:
        x = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']
        return train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None, None