import pandas as pd
from sklearn.impute import SimpleImputer
import featuretools as ft
import re
from src.utilities import create_entity

def main_preprocess(data):

    data['clinical'] = parse_cytogenetics_column(data['clinical'].reset_index(drop=True), column_name='CYTOGENETICS')

    X, features_defs = ft.dfs(entityset=data, target_dataframe_name="clinical")
    X = process_categories(X, method="del")

    return X

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

def parse_cytogenetics(cyto_str):
    """
    Prend en entrée une chaîne décrivant la cytogénétique (ex : "46,xy,del(3)(q26q27)[15]/46,xy[5]")
    et retourne un dictionnaire de features utiles.
    """
    
    if pd.isna(cyto_str):
        # Si la valeur est manquante, on peut retourner un dictionnaire vide
        return {
            'num_subclones': 0, 'sex': None, 'avg_chromosomes': None,
            'total_mitoses': 0, 'num_translocations': 0, 'num_deletions': 0,
            'num_inversions': 0, 'num_duplications': 0, 'num_additions': 0,
            'num_monosomies': 0, 'num_trisomies': 0, 'complexity_score': 0
        }

    # Séparer la description en sous-clones (split sur '/')
    subclones = cyto_str.split('/')
    
    # On initialise des compteurs globaux
    total_mitoses = 0
    total_translocations = 0
    total_deletions = 0
    total_inversions = 0
    total_duplications = 0
    total_additions = 0
    total_monosomies = 0
    total_trisomies = 0
    
    # Liste pour stocker le nombre de chromosomes détectés dans chaque sous-clone
    clone_chromosome_numbers = []
    
    # Pour détecter le sexe
    sex = None
    
    for clone in subclones:
        # Exemple de clone : "46,xy,del(3)(q26q27)[15]" ou "46,xx[10]"
        
        # Extraire le nombre de mitoses dans [x], s'il existe
        mitoses = 0
        mitoses_match = re.search(r'\[(\d+)\]', clone)
        if mitoses_match:
            mitoses = int(mitoses_match.group(1))
        total_mitoses += mitoses
        
        # Retirer la partie [x]
        clone_clean = re.sub(r'\[\d+\]', '', clone)
        
        # Split par virgule
        parts = clone_clean.split(',')
        
        for p in parts:
            p = p.strip().lower()
            
            # Nombre de chromosomes ?
            if re.match(r'^\d+$', p):
                # ex : "46"
                clone_chromosome_numbers.append(int(p))
            
            # Détecter le sexe
            if "xy" in p:
                if sex is None:
                    sex = 1
            elif "xx" in p:
                if sex is None:
                    sex = 0

            
            # Rechercher anomalies
            if re.search(r't\(\d+;\d+\)', p):
                total_translocations += 1
            if "del(" in p:
                total_deletions += 1
            if "inv(" in p:
                total_inversions += 1
            if "dup(" in p:
                total_duplications += 1
            if "add(" in p:
                total_additions += 1
            
            # Trisomies / monosomies notées +7, -5, etc.
            plus_match = re.search(r'\+(\d+)', p)
            minus_match = re.search(r'\-(\d+)', p)
            if plus_match:
                total_trisomies += 1
            if minus_match:
                total_monosomies += 1
    
    # Nombre de sous-clones
    num_subclones = len(subclones)
    
    # Moyenne (non pondérée) du nombre de chromosomes
    if len(clone_chromosome_numbers) > 0:
        avg_chromosomes = sum(clone_chromosome_numbers) / len(clone_chromosome_numbers)
    else:
        avg_chromosomes = None
    
    complexity_score = (total_translocations + total_deletions + total_inversions +
                        total_duplications + total_additions)

    return {
        'num_subclones': num_subclones,
        'sex': sex if sex else -1,
        'avg_chromosomes': avg_chromosomes,
        'total_mitoses': total_mitoses,
        'num_translocations': total_translocations,
        'num_deletions': total_deletions,
        'num_inversions': total_inversions,
        'num_duplications': total_duplications,
        'num_additions': total_additions,
        'num_monosomies': total_monosomies,
        'num_trisomies': total_trisomies,
        'complexity_score': complexity_score
    }


def parse_cytogenetics_column(df, column_name='CYTOGENETICS'):
    """
    Prend un DataFrame `df` et le nom de la colonne cytogénétique `column_name`.
    Retourne un nouveau DataFrame comprenant les features extraites.
    """
    # Appliquer la fonction parse_cytogenetics à chaque ligne
    parsed_series = df[column_name].apply(parse_cytogenetics)
    
    # Convertir la série de dictionnaires en DataFrame
    parsed_df = pd.json_normalize(parsed_series)
    
    # Concaténer avec le DataFrame d'origine (sans dupliquer la colonne de base si vous voulez la garder)
    # Si vous préférez garder la colonne CYTOGENETICS, ne la supprimez pas.
    final_df = pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)
    
    return final_df