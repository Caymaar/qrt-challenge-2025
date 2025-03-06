from sksurv.metrics import concordance_index_ipcw
import numpy as np
import pandas as pd
from datetime import datetime

def convert_survival_data(structured_array):
    """
    Convertit un tableau structuré de données de survie au format attendu.
    
    Args:
        structured_array: numpy.ndarray structuré avec ('event', '?') et ('time', '<f8')
    
    Returns:
        tuple: (times, events) où times est un array de float32 et events est un array de int32
    """
    import numpy as np
    
    # Extraction des temps et événements
    times = structured_array['time'].astype(np.float32)
    events = structured_array['event'].astype(np.int32)
    
    return (times, events)

def convert_float32(X_train, X_test, X_eval):
    """
    Convertit les données
    """

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_eval = X_eval.astype('float32')

    return (X_train, X_test, X_eval)

def score_method_deep(model, X_train_deep, X_test_deep, y_train, y_test, reverse=False):
    X_train_deep = X_train_deep.astype(np.float32)
    X_test_deep = X_test_deep.astype(np.float32)
    cindex_train = concordance_index_ipcw(y_train, y_train, model.predict(X_train_deep).flatten() if not reverse else -model.predict(X_train_deep), tau=7)[0]
    cindex_test = concordance_index_ipcw(y_train, y_test, model.predict(X_test_deep).flatten() if not reverse else -model.predict(X_test_deep), tau=7)[0]
    print(f"{model.__class__.__name__} Model Concordance Index IPCW on train: {cindex_train:.3f}")
    print(f"{model.__class__.__name__} Model Concordance Index IPCW on test: {cindex_test:.3f}")
    return f"score_{cindex_train:.3f}_{cindex_test:.3f}"

def predict_and_save_deep(X_eval, model, method="featuretools"):
    df_eval = pd.read_csv("data/X_test/clinical_test.csv")
    prediction_on_test_set = -model.predict(X_eval) if model.__class__.__name__ == "Booster" else model.predict(X_eval).flatten()
    submission = pd.Series(prediction_on_test_set, index=df_eval['ID'], name='risk_score')
    submission.to_csv(f'output/{model.__class__.__name__}_submission_{method}_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
