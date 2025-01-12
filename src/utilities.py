import pandas as pd
import featuretools as ft
from sksurv.util import Surv
from datetime import datetime

def load_target():
    target_df = pd.read_csv("data/target_train.csv")
    target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)
    target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')
    target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)

    return target_df

def create_entity():

    clinical_train = pd.read_csv("data/X_train/clinical_train.csv")
    molecular_train = pd.read_csv("data/X_train/molecular_train.csv")

    clinical_test = pd.read_csv("data/X_test/clinical_test.csv")
    molecular_test = pd.read_csv("data/X_test/molecular_test.csv")

    clinical = pd.concat([clinical_train, clinical_test]).reset_index(drop=True)
    molecular = pd.concat([molecular_train, molecular_test]).reset_index(drop=True)

    es = ft.EntitySet(id="data")

    es = es.add_dataframe(
        dataframe_name='clinical',
        dataframe=clinical,
        index='ID' 
    )

    es = es.add_dataframe(
        dataframe_name='molecular',
        dataframe=molecular,
        index='index'   
    )

    es.add_relationship(
        parent_dataframe_name='clinical',
        parent_column_name='ID',
        child_dataframe_name='molecular',
        child_column_name='ID'
    )

    return es

def split_data(X_data):

    df_eval = pd.read_csv("data/X_test/clinical_test.csv")
    X_eval = X_data.loc[X_data.index.isin(df_eval['ID'])]

    target = load_target()

    # Create the survival data format
    X_data = X_data.loc[X_data.index.isin(target['ID'])]
    y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target)

    return X_data, X_eval, y

def predict_and_save(X_eval, model, method="featuretools"):
    df_eval = pd.read_csv("data/X_test/clinical_test.csv")
    prediction_on_test_set = -model.predict(X_eval) if model.__class__.__name__ == "Booster" else model.predict(X_eval)
    submission = pd.Series(prediction_on_test_set, index=df_eval['ID'], name='risk_score')
    submission.to_csv(f'output/{model.__class__.__name__}_submission_{method}_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')

