import pandas as pd
from sklearn.impute import SimpleImputer
import re

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


def preprocess_caryotype(caryotype):
    """
    Prend en entrée une chaîne représentant un caryotype
    et retourne un dictionnaire avec des informations prétraitées.
    """

    # Initialisation des résultats
    result = {
        "sex": None,  # Sexe (Homme ou Femme)
        "total_chromosomes": 0,  # Nombre total de chromosomes
        "anomalies": [],  # Liste des anomalies détectées
        "mosaicism": False,  # Mosaïcisme présent ou non
    }

    # Identifier le sexe
    if "xy" in caryotype:
        result["sex"] = "Homme"
    elif "xx" in caryotype:
        result["sex"] = "Femme"
    else:
        result["sex"] = "Inconnu"

    # Extraire le nombre total de chromosomes
    match_total = re.search(r"(\d+),", caryotype)
    if match_total:
        result["total_chromosomes"] = int(match_total.group(1))

    # Identifier les anomalies (par exemple, délétions, duplications, translocations)
    anomalies = re.findall(r"(del|dup|t|inv)\((\d+)\)(\(q\d+.*?\))?", caryotype)
    for anomaly in anomalies:
        anomaly_type, chromosome, details = anomaly
        result["anomalies"].append({
            "type": anomaly_type,
            "chromosome": int(chromosome),
            "details": details if details else "NA",
        })

    # Détecter le mosaïcisme
    if "/" in caryotype:
        result["mosaicism"] = True

    return result