import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import featuretools as ft
import re
from src.utilities import create_entity

def main_preprocess(data, clinical_process, molecular_process, merge_process):

    # data = create_entity()
    # clinical_process = ["CYTOGENETICS"]
    # molecular_process = ["GENE"]
    # merge_process = "featureto"

    if "CYTOGENETICS" in clinical_process:
        data['clinical'] = parse_cytogenetics_column(data['clinical'].reset_index(drop=True), column_name='CYTOGENETICS')

    if "featuretools" == merge_process:
        X, features_defs = ft.dfs(entityset=data, target_dataframe_name="clinical")
    else:
        X = data['clinical'].set_index('ID')

    if "GENE" in molecular_process:
        gene = create_one_hot(data['molecular'], id_col='ID', ref_col='GENE', min_count=50, rare_label='gene_other')
        X = gene.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    if "EFFECT" in molecular_process:
        effect = create_one_hot(data['molecular'], id_col='ID', ref_col='EFFECT', min_count=0, rare_label='effect_other')
        X = effect.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    if "REF" in molecular_process:
        ref = count_bases_per_id(data['molecular'], id_col='ID', ref_col='REF')
        X = ref.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    if "ALT" in molecular_process:
        alt = count_bases_per_id(data['molecular'], id_col='ID', ref_col='ALT')
        X = alt.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    X = add_ratio_column(X, 'HB', 'PLT')
    
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
                    sex = 1.0
            elif "xx" in p:
                if sex is None:
                    sex = 0.0

            
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

    final_df['sex'].fillna(0, inplace=True)

    return final_df

def create_one_hot(
    df, 
    id_col='ID', 
    ref_col='GENE', 
    min_count=10,
    rare_label='gene_other'
):
    """
    Pour chaque 'ID' unique, crée un One-Hot Encoding des gènes.
    Les gènes qui apparaissent moins de min_count fois dans df
    sont regroupés sous la même catégorie (rare_label).
    
    Paramètres
    ----------
    df : DataFrame
        Contient au moins les colonnes [id_col, gene_col].
    id_col : str
        Nom de la colonne identifiant unique du patient (ID).
    gene_col : str
        Nom de la colonne contenant le gène.
    min_count : int
        Seuil de fréquence minimum en dessous duquel un gène est classé dans rare_label.
    rare_label : str
        Nom de la catégorie pour regrouper les gènes rares.
    
    Retour
    ------
    pivoted_df : DataFrame
        Tableau pivoté de dimension [nb_ID x nb_gènes+1], 
        où chaque gène (ou GENE_AUTRE) est codé 0/1.
        L'index est remis dans une colonne 'ID'.
    """

    # 1) Compter la fréquence de chaque gène
    freq = df[ref_col].value_counts()

    # 2) Identifier les gènes rares
    rare_genes = freq[freq < min_count].index  # index -> liste des gènes
    # => Ce sont tous les gènes qui apparaissent < min_count fois

    # 3) Créer une colonne "gene_aggreg" 
    #    qui remplace les gènes rares par "GENE_AUTRE"
    df[f'{ref_col.lower()}_aggreg'] = df[ref_col].apply(
        lambda g: rare_label if g in rare_genes else g
    )

    # 4) Réaliser le pivot / crosstab
    #    Pour chaque ID, on indique 1 si le gène (agrégé) est présent, 0 sinon.
    pivoted = pd.crosstab(df[id_col], df[f'{ref_col.lower()}_aggreg'])

    # Facultatif : renommer les colonnes pour y faire apparaître "gene_"
    pivoted.columns = [f"{ref_col.lower()}_{col}" for col in pivoted.columns]

    # Remettre l'index (ID) comme colonne
    pivoted.reset_index(inplace=True)

    return pivoted


def count_bases_per_id(df, id_col='ID', ref_col='REF'):
    """
    Agrège les chaînes de caractères de la colonne `ref_col` par `id_col`,
    puis compte le nombre de A, G, C, T. Retourne un DataFrame 
    avec les colonnes [id_col, ref_A, ref_G, ref_C, ref_T].
    
    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame contenant au moins `id_col` et `ref_col`.
    id_col : str
        Nom de la colonne identifiant (ex. 'ID').
    ref_col : str
        Nom de la colonne contenant des lettres (ex. f'{col.lower()}').

    Retour
    ------
    pd.DataFrame
        Un DataFrame de taille [nombre_unique_de_ID x 5], 
        avec les colonnes: [id_col, f'{col.lower()}_A', f'{col.lower()}_G', f'{col.lower()}_C', f'{col.lower()}_T'].
    """
    
    # 1) Copie de sécurité pour éviter de modifier df directement
    df_copy = df.copy()
    
    # 2) Remplacer les NaN par des chaînes vides dans la colonne `ref_col`
    df_copy[ref_col] = df_copy[ref_col].astype(str).fillna('')
    
    # 3) Grouper par `id_col`, puis concaténer toutes les chaînes en une seule
    grouped_strings = df_copy.groupby(id_col)[ref_col].apply(lambda x: ''.join(x))
    
    # 4) Définir une fonction pour compter les occurrences de A, G, C, T
    def count_bases(sequence):
        return pd.Series({
            f'{ref_col.lower()}_A': sequence.count('A'),
            f'{ref_col.lower()}_G': sequence.count('G'),
            f'{ref_col.lower()}_C': sequence.count('C'),
            f'{ref_col.lower()}_T': sequence.count('T')
        })
    
    # 5) Appliquer cette fonction de comptage sur chaque séquence agrégée
    base_counts = grouped_strings.apply(count_bases).reset_index()
    
    # 6) Vous pouvez simplement retourner base_counts,
    #    ou si vous voulez être certain de conserver tous les ID originaux
    #    (y compris ceux n'ayant aucune entrée dans `ref_col`),
    #    on merge avec la liste des ID distincts :
    distinct_ids = df_copy[[id_col]].drop_duplicates()
    result = distinct_ids.merge(base_counts, on=id_col, how='left')
    
    # 7) Optionnel: remplir les NaN (pour les ID sans aucune base) par 0
    result[[f'{ref_col.lower()}_A',f'{ref_col.lower()}_G',f'{ref_col.lower()}_C',f'{ref_col.lower()}_T']] = result[[f'{ref_col.lower()}_A',f'{ref_col.lower()}_G',f'{ref_col.lower()}_C',f'{ref_col.lower()}_T']].fillna(0)
    
    return result

def add_ratio_column(df, col_num, col_den, new_col_name=None):
    """
    Crée dans `df` une nouvelle colonne représentant le ratio col_num / col_den.
    Gère les cas où col_den est 0 ou NaN en attribuant NaN au ratio.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à enrichir.
    col_num : str
        Nom de la colonne faisant office de numérateur.
    col_den : str
        Nom de la colonne faisant office de dénominateur.
    new_col_name : str, optionnel
        Nom de la nouvelle colonne. S'il n'est pas fourni,
        le nom sera 'ratio_{col_num}_{col_den}'.

    Retour
    ------
    pd.DataFrame
        Le même DataFrame (modifié en place) avec la colonne ratio ajoutée.
    """
    if new_col_name is None:
        new_col_name = f"ratio_{col_num}_{col_den}"

    # Copie de sécurité optionnelle (décommentez si vous ne voulez pas modifier le df original)
    # df = df.copy()

    # Pour gérer les cas où le dénominateur est NaN ou 0,
    # on calcule le ratio dans une série, puis on l'assigne.
    ratio_series = pd.Series(np.nan, index=df.index)

    # On peut définir une condition qui identifie où col_den est non-nul, non-NaN
    valid_den = (df[col_den].notna()) & (df[col_den] != 0)

    # Pour les lignes valides, on calcule le ratio
    ratio_series[valid_den] = df.loc[valid_den, col_num] / df.loc[valid_den, col_den]

    # On ajoute la série au DataFrame
    df[new_col_name] = ratio_series

    return df