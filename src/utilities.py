import pandas as pd
import featuretools as ft

def load_target():
    target_df = pd.read_csv("data/target_train.csv")
    target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)
    target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')
    target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)

    return target_df

def create_entity(status):

    clinical = pd.read_csv(f"data/X_{status}/clinical_{status}.csv")
    molecular = pd.read_csv(f"data/X_{status}/molecular_{status}.csv").reset_index()
    
    es = ft.EntitySet(id=status)

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

def predict_and_save(X_eval, model, method="featuretools"):
    df_eval = pd.read_csv("data/X_test/clinical_test.csv")
    prediction_on_test_set = model.predict(X_eval)
    submission = pd.Series(prediction_on_test_set, index=df_eval['ID'], name='risk_score')
    submission.to_csv(f'output/{model.__class__.__name__}_submission_{method}.csv')

