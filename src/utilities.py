import pandas as pd
import featuretools as ft
from sksurv.util import Surv
from datetime import datetime
from sksurv.metrics import concordance_index_ipcw

def load_target():
    target_df = pd.read_csv("data/target_train.csv")
    target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)
    target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')
    target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)

    return target_df

def split_data(X_data, reverse=False):

    df_eval = pd.read_csv("data/X_test/clinical_test.csv")
    X_eval = X_data.loc[X_data.index.isin(df_eval['ID'])]

    target = load_target()

    # Create the survival data format
    X_data = X_data.loc[X_data.index.isin(target['ID'])]

    #Rename columns of target
    target.rename(columns={'OS_STATUS': 'event', 'OS_YEARS': 'time'}, inplace=True)
    

    if reverse:
        y = Surv.from_dataframe('time', 'event', target)
    else:
        y = Surv.from_dataframe('event', 'time', target)
    return X_data, X_eval, y

def get_method_name(key, params, model=False):
    if key not in params:
        return ""
    if model:
        return str(key) + "_" + "_".join([str(params[key][param]) for param in params[key].keys()])
    if isinstance(params[key], list):
        return str(key) + "_" + "_".join(params[key]).replace("/", "divby")
    else:
        return str(key) + "_" + str(params[key])
    
def score_method(model, X_train, X_test, y_train, y_test, reverse=False):
    cindex_train = concordance_index_ipcw(y_train, y_train, model.predict(X_train) if not reverse else -model.predict(X_train), tau=7)[0]
    cindex_test = concordance_index_ipcw(y_train, y_test, model.predict(X_test) if not reverse else -model.predict(X_test), tau=7)[0]
    print(f"{model.__class__.__name__} Model Concordance Index IPCW on train: {cindex_train:.3f}")
    print(f"{model.__class__.__name__} Model Concordance Index IPCW on test: {cindex_test:.3f}")
    return f"score_{cindex_train:.3f}_{cindex_test:.3f}"

def predict_and_save(X_eval, model, method="featuretools"):
    df_eval = pd.read_csv("data/X_test/clinical_test.csv")
    prediction_on_test_set = -model.predict(X_eval) if model.__class__.__name__ == "Booster" else model.predict(X_eval)
    submission = pd.Series(prediction_on_test_set, index=df_eval['ID'], name='risk_score')
    submission.to_csv(f'output/{model.__class__.__name__}_submission_{method}_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')

