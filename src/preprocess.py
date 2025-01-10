import pandas as pd
from sklearn.impute import SimpleImputer

def process_categories(df, method="del"):
    categorical_columns = df.select_dtypes(include=['category']).columns

    if method == "del":
        return df.drop(categorical_columns, axis=1)
    
    if method == "dummies":
        return pd.get_dummies(df, columns=categorical_columns)

def process_missing_values(X_train, X_test, X_eval, method="impute", strategy="median"):
    if method == "del":
        return X_train.dropna(), X_test.dropna(), X_eval.dropna()
    
    if method == "impute":
        imputer = SimpleImputer(strategy=strategy)
        return imputer.fit_transform(X_train), imputer.transform(X_test), imputer.transform(X_eval)
