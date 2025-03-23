import pandas as pd
import featuretools as ft
from sksurv.util import Surv
from datetime import datetime
from sksurv.metrics import concordance_index_ipcw
import hashlib
import os

ARCHIVE_FILE = "archive.xlsx"

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

def predict_and_save(X_eval, model, name):
    df_eval = pd.read_csv("data/X_test/clinical_test.csv")
    prediction_on_test_set = -model.predict(X_eval) if model.__class__.__name__ == "Booster" else model.predict(X_eval)
    submission = pd.Series(prediction_on_test_set, index=df_eval['ID'], name='risk_score')
    submission.to_csv(f'output/{name}.csv')

def run_model(model_key, X_train, X_test, y_train, y_test, X_eval, method_formatter):
    """
    Entraîne un modèle, calcule son score et effectue les étapes optionnelles de sauvegarde
    et de génération de rapport SHAP.

    :param model_key: Clé pour accéder au modèle dans GLOBAL (ex: "cox", "xgb", "rsf").
    :param X_train: Données d'entraînement.
    :param X_test: Données de test.
    :param y_train: Cibles d'entraînement.
    :param y_test: Cibles de test.
    :param X_eval: Données d'évaluation pour les prédictions.
    :param method_formatter: Fonction lambda ou callable qui prend le score et retourne la chaîne
                             de méthode (ex: lambda score: f"{size_method}-{score}-...").
    :return: Le modèle entraîné et son score.
    """
    model_cfg = GLOBAL[model_key]
    
    # Si le modèle n'est pas à exécuter, on retourne None
    if not model_cfg.get("run", False):
        return None, None

    # Récupération du modèle initialisé dans GLOBAL
    model = model_cfg.get("model")
    
    # Entraînement et évaluation
    model.fit(X_train, y_train)
    model_score = score_method(model, X_train, X_test, y_train, y_test)
    
    # Formatter le nom/méthode en fonction du score et d'autres paramètres
    method_name = method_formatter(model_score) if callable(method_formatter) else method_formatter
    
    # Sauvegarde des prédictions si nécessaire
    if model_cfg.get("save", False):
        predict_and_save(X_eval, model, method=method_name)
    
    # Génération du rapport SHAP si activé
    if model_cfg.get("shap", False):
        from src.report import ShapReport
        # S'assurer que X_train est un DataFrame
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)
        report = ShapReport(model=model, X_train=X_train, predict_function=model.predict)
        report.generate_report(output_html=f"report/shap/shap_{model.__class__.__name__}_{method_name}.html")
    
    return model, model_score


def generate_token(long_string, token_length=16):
    """
    Génère un token en prenant les 'token_length' premiers caractères du hash MD5 de la chaîne.
    Par défaut, token_length est fixé à 16, mais vous pouvez l'augmenter selon vos besoins.
    """
    hash_object = hashlib.md5(long_string.encode('utf-8'))
    token = hash_object.hexdigest()[:token_length]
    return token

def add_to_archive(token, long_string):
    """
    Ajoute la correspondance token <-> nom de fichier dans l'archive Excel.
    Si l'archive existe déjà, elle est lue et mise à jour.
    """
    if os.path.exists(ARCHIVE_FILE):
        df = pd.read_excel(ARCHIVE_FILE)
    else:
        df = pd.DataFrame(columns=["token", "filename"])
    
    # Vérifier si le token existe déjà dans l'archive
    if token in df['token'].values:
        existing = df.loc[df['token'] == token, 'filename'].iloc[0]
        if existing != long_string:
            print("Attention : collision détectée pour le token", token)
    else:
        new_row = pd.DataFrame([[token, long_string]], columns=["token", "filename"])
        df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_excel(ARCHIVE_FILE, index=False)

def encode_filename(long_string, token_length=16):
    """
    Encode le nom de fichier long en générant un token de longueur spécifiée et en l'ajoutant à l'archive.
    """
    token = generate_token(long_string, token_length=token_length)
    add_to_archive(token, long_string)
    print(f"Token {token} généré et ajouté à l'archive.")
    return token

def decode_filename(token):
    """
    Recherche dans l'archive le nom de fichier associé au token donné.
    """
    if os.path.exists(ARCHIVE_FILE):
        df = pd.read_excel(ARCHIVE_FILE)
        if token in df['token'].values:
            return df.loc[df['token'] == token, 'filename'].iloc[0]
        else:
            print("Token non trouvé dans l'archive.")
            return None
    else:
        print("L'archive n'existe pas.")
        return None
    
def rename_column(col):
    # Remplacer les fonctions et supprimer "molecular." pour réduire la redondance
    col = col.replace("SKEW(molecular.", "skw_")
    col = col.replace("STD(molecular.", "std_")
    col = col.replace("SUM(molecular.", "sum_")
    col = col.replace("MAX(molecular.", "max_")
    col = col.replace("MIN(molecular.", "min_")
    col = col.replace("MEAN(molecular.", "mean_")
    col = col.replace("COUNT(molecular.", "count_")
    col = col.replace(")", "")
    return col

def prepare_EDA(cols, X_train, y_train):
    # Appliquer le renommage à toutes les colonnes du DataFrame
    renamed_cols = [rename_column(col) for col in cols]

    df_analyze = pd.concat([pd.DataFrame(X_train, columns=renamed_cols), pd.DataFrame(y_train, columns=["event", "time"])],axis=1)
    bool_cols = df_analyze.select_dtypes(include=['bool']).columns
    df_analyze[bool_cols] = df_analyze[bool_cols].astype(int)

    return df_analyze