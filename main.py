from src.utilities import predict_and_save, split_data, get_method_name, score_method, encode_filename
from src.preprocess import process_missing_values, main_preprocess, create_entity
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
import lightgbm as lgb
import pandas as pd
from datetime import datetime

import warnings
import logging

# Régler le logger de Featuretools au niveau ERROR
logging.getLogger('featuretools.entityset').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",message=".*Ill-conditioned matrix.*")

GLOBAL = {
    "cox": {"run": False, "save":False, "shap": False},
    "xgb": {"run": True, "save": False, "shap": True},
    "lgbm": {"run": False, "save":False, "shap": False},
    "rsf": {"run": False, "save":False, "shap": False}
}

PARAMS = {
    "EDA" : False, 
    "size": 0.7,
    "impute": {"strategy": "median", "sex": False},
    #"outliers": {"threshold": 0.01, "multiplier": 1.5},
    "clinical": ["CYTOGENETICS"],#["CYTOGENETICS"], # Possible: ["CYTOGENETICS", "HB/PLT", "logMONOCYTES", "logWBC", "logANC"] ["BM_BLAST+WBC", "BM_BLAST/HB", "HB*PLT", "HB/num_trisomies"]
    "molecular": ["GENE"],#["END-START"], # Possible: ["GENE", "EFFECT", "ALT", "REF", "END-START"]
    "merge": ["featuretools", "gpt"], # Possible: ["featuretools", "gpt"]
    "additional": [
        #['cadd', 'phred'],
        # ['cadd', 'rawscore'],
        # # ['cadd', 'consequence'],
        # # ['cadd', 'bstatistic'],
        # # ['cadd', 'gerp', 'n'],
        # ['cadd', 'phast_cons', 'mammalian'],
        # ['cadd', 'phylop', 'mammalian'],
        # ['snpeff', 'putative_impact'],
        # # ['snpeff', 'rank'],
        # # ['snpeff', 'total'],
         #['cadd', 'exon'],
        # # ['cadd', 'cds', 'rel_cds_pos']
        ],
    "xgb": {
        'loss': 'coxph',
        'max_depth': 2,
        'learning_rate': 0.05,
        'n_estimators': 335,
        'subsample': 0.55,
        'max_features': "sqrt",
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0,
        'min_impurity_decrease': 0,
        'dropout_rate': 0,
        'warm_start': False,
        'ccp_alpha': 0,
        'random_state': 126
    },
    "lgbm": {
        'max_depth': 2,
        'learning_rate': 0.05,
        'verbose': 0
    },
    "rsf": {
    'n_estimators':200,  # Nombre d'arbres dans la forêt
    'max_depth':None,
    'min_samples_split':50,  # Nombre minimum d'échantillons requis pour splitter un nœud
    'min_samples_leaf':20,  # Nombre minimum d'échantillons par feuille
    'max_features':'sqrt',  # Sélection aléatoire des features
    'n_jobs':-1,  # Utilisation de tous les cœurs disponibles
    }
}


data = create_entity(PARAMS)
data = main_preprocess(data, PARAMS)
X, X_eval, y = split_data(data)

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - PARAMS['size']), random_state=42)
X_train, X_test, X_eval = process_missing_values(X_train, X_test, X_eval, X.columns, **PARAMS["impute"])

if PARAMS["EDA"]:
    from src.report import EDAReport
    from src.utilities import prepare_EDA

    df_analyze = prepare_EDA(X.columns, X_train, y_train)

    ede = EDAReport(df_analyze, target_variables=["event", "time"])
    ede.generate_report(max_rows=100, max_cols=10)

##############################################
# Define the methods used for training
##############################################

size_method = get_method_name("size", PARAMS)
clinical_method = get_method_name("clinical", PARAMS)
molecular_method = get_method_name("molecular", PARAMS)
merge_method = get_method_name("merge", PARAMS)

for model in GLOBAL.keys():
    if GLOBAL[model]["run"]:
        print(f"Running {model} with method: {size_method}-{clinical_method}-{molecular_method}-{merge_method}")

##############################################
# Fit a CoxPH model
##############################################

if GLOBAL["cox"]["run"]:

    cox = CoxPHSurvivalAnalysis()
    cox.fit(X_train, y_train)
    cox_score_method = score_method(cox, X_train, X_test, y_train, y_test)

    if GLOBAL["cox"]["save"] or GLOBAL["cox"]["shap"]:
        method = f"{size_method}-{cox_score_method}-{clinical_method}-{molecular_method}-{merge_method}"
        name = f'{cox.__class__.__name__}_submission_{method}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        token = encode_filename(name, token_length=40)

    if GLOBAL["cox"]["save"]:
        predict_and_save(X_eval, cox, name=token)

    if GLOBAL["cox"]["shap"]:
        from src.report import ShapReport

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)

        report_cox = ShapReport(model=cox, X_train=X_train, predict_function=cox.predict)
        report_cox.generate_report(output_file=f"report/shap/shap_{token}.html")


##############################################
# Fit a Gradient Boosting model
##############################################

if GLOBAL["xgb"]["run"]:
    xgb_params_method = "_".join([(str(key) + "=" + str(PARAMS['xgb'][key])) for key in PARAMS['xgb'].keys()])

    xgb = GradientBoostingSurvivalAnalysis(**PARAMS['xgb'])
    xgb.fit(X_train, y_train)
    xgboost_score_method = score_method(xgb, X_train, X_test, y_train, y_test)

    if GLOBAL["xgb"]["save"] or GLOBAL["xgb"]["shap"]:
        method = f"{size_method}-{xgboost_score_method}--{molecular_method}-{merge_method}-{xgb_params_method}"
        name = f'{xgb.__class__.__name__}_submission_{method}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

        token = encode_filename(name, token_length=40)

    if GLOBAL["xgb"]["save"]:
        predict_and_save(X_eval, xgb, name=token)

    if GLOBAL["xgb"]["shap"]:
        from src.report import ShapReport

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)

        report_xgb = ShapReport(model=xgb, X_train=X_train, predict_function=xgb.predict)
        report_xgb.generate_report(output_file=f"report/shap/shap_{token}.html")


##############################################
# Fit a LightGBM model
##############################################

if GLOBAL["lgbm"]["run"]:
    lgbm_params_method = "_".join([f"{key}={PARAMS['lgbm'][key]}" for key in PARAMS['lgbm'].keys()])

    train_dataset = lgb.Dataset(X_train, label=y_train['time'])
    model_lgb = lgb.train(params=PARAMS['lgbm'], train_set=train_dataset)
    lightgbm_score_method = score_method(model_lgb, X_train, X_test, y_train, y_test, reverse=True)

    if GLOBAL["lgbm"]["save"] or GLOBAL["lgbm"]["shap"]:
        method = f"{size_method}-{lightgbm_score_method}-{clinical_method}-{molecular_method}-{merge_method}-{lgbm_params_method}"
        name = f'{model_lgb.__class__.__name__}_submission_{method}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        token = encode_filename(name, token_length=40)

    if GLOBAL["lgbm"]["save"]:
        predict_and_save(X_eval, model_lgb, name=token)

    if GLOBAL["lgbm"]["shap"]:
        from src.report import ShapReport

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)

        report_lgbm = ShapReport(model=model_lgb, X_train=X_train, predict_function=model_lgb.predict)
        report_lgbm.generate_report(output_file=f"report/shap/shap_{token}.html")

##############################################
# Fit a Random Survival Forest model
##############################################

if GLOBAL["rsf"]["run"]:
    rsf_params_method = "_".join([(str(key) + "=" + str(PARAMS['rsf'][key])) for key in PARAMS['rsf'].keys()])

    rsf = RandomSurvivalForest(**PARAMS["rsf"], random_state=42)
    rsf.fit(X_train, y_train)
    rsf_score_method = score_method(rsf, X_train, X_test, y_train, y_test)

    if GLOBAL["rsf"]["save"] or GLOBAL["rsf"]["shap"]:
        method = f"{size_method}-{rsf_score_method}-{clinical_method}-{molecular_method}-{merge_method}-{rsf_params_method}"
        name = f'{rsf.__class__.__name__}_submission_{method}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        token = encode_filename(name, token_length=40)

    if GLOBAL["rsf"]["save"]:
        predict_and_save(X_eval, rsf, name=token)

    if GLOBAL["rsf"]["shap"]:
        from src.report import ShapReport

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)

        report_rsf = ShapReport(model=rsf, X_train=X_train, predict_function=rsf.predict)
        report_rsf.generate_report(output_file=f"report/shap/shap_{token}.html")