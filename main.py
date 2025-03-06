from src.utilities import predict_and_save, split_data, get_method_name, score_method
from src.preprocess import process_missing_values, main_preprocess, create_entity
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
import lightgbm as lgb
import pandas as pd

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
    "xgb": {"run": True, "save":True, "shap": False},
    "lgbm": {"run": False, "save":False, "shap": False},
    "rsf": {"run": False, "save":False, "shap": False}
}

PARAMS = {
    "size": 0.7,
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
        'max_depth': 2,
        'learning_rate': 0.05,
        'n_estimators': 450,
        'subsample': 0.55,
        'max_features': 'sqrt',
        'random_state': 26
    },
    "lgbm": {
        'max_depth': 2,
        'learning_rate': 0.05,
        'verbose': 0
    },
    "rsf": {
    'n_estimators':300,  # Nombre d'arbres dans la forêt
    'max_depth':2,
    #'min_samples_split':60,  # Nombre minimum d'échantillons requis pour splitter un nœud
    #'min_samples_leaf':40,  # Nombre minimum d'échantillons par feuille
    'max_features':None,  # Sélection aléatoire des features
    'n_jobs':-1,  # Utilisation de tous les cœurs disponibles
    }
}


data = create_entity(PARAMS)
data = main_preprocess(data, PARAMS)
X, X_eval, y = split_data(data)

# Check if there are any columns that are not float or int in X
print(X.columns)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - PARAMS['size']), random_state=42)
X_train, X_test, X_eval = process_missing_values(X_train, X_test, X_eval, method="impute", strategy="median")
# Ensure all columns in X_train are either float or int

##############################################
# Define the methods used for training
##############################################

size_method = get_method_name("size", PARAMS)
clinical_method = get_method_name("clinical", PARAMS)
molecular_method = get_method_name("molecular", PARAMS)
merge_method = get_method_name("merge", PARAMS)

##############################################
# Fit a CoxPH model
##############################################

if GLOBAL["cox"]["run"]:
    # Initialize and train the Cox Proportional Hazards model
    cox = CoxPHSurvivalAnalysis()    
    cox.fit(X_train, y_train)
    cox_score_method = score_method(cox, X_train, X_test, y_train, y_test)

    # Predict and save the results
    if GLOBAL["cox"]["save"]:
        predict_and_save(X_eval, cox, method=f"{size_method}-{cox_score_method}-{clinical_method}-{molecular_method}-{merge_method}")

    if GLOBAL["cox"]["shap"]:
        from src.report import ShapReport

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)

        report_cox = ShapReport(model=cox, X_train=X_train, predict_function=cox.predict)
        report_cox.generate_report(output_html=f"report/shap/shap_{cox.__class__.__name__}_{size_method}-{cox_score_method}-{clinical_method}-{molecular_method}-{merge_method}.html")



##############################################
# Fit a Gradient Boosting model
##############################################

if GLOBAL["xgb"]["run"]:
    xgb_params_method = "_".join([(str(key) + "=" + str(PARAMS['xgb'][key])) for key in PARAMS['xgb'].keys()])

    xgb = GradientBoostingSurvivalAnalysis(**PARAMS['xgb'])
    xgb.fit(X_train, y_train)
    xgboost_score_method = score_method(xgb, X_train, X_test, y_train, y_test)

    if GLOBAL["xgb"]["save"]:
        predict_and_save(X_eval, xgb, method=f"{size_method}-{xgboost_score_method}--{molecular_method}-{merge_method}-{xgb_params_method}")

    if GLOBAL["xgb"]["shap"]:
        from src.report import ShapReport

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)

        report_xgb = ShapReport(model=xgb, X_train=X_train, predict_function=xgb.predict)
        report_xgb.generate_report(output_html=f"report/shap/shap_{xgb.__class__.__name__}_{size_method}-{xgboost_score_method}--{molecular_method}-{merge_method}-{xgb_params_method}.html")


##############################################
# Fit a LightGBM model
##############################################

# X_train_lgb = X_train  # Features for training
# y_train_transformed = y_train['time']

# # Create LightGBM dataset
# train_dataset = lgb.Dataset(X_train_lgb, label=y_train_transformed)

# # Train the LightGBM model
# model = lgb.train(params=PARAMS['lgbm'], train_set=train_dataset)

# # Evaluate the model using Concordance Index IPCW
# train_ci_ipcw = concordance_index_ipcw(y_train, y_train, -model.predict(X_train), tau=7)[0]
# test_ci_ipcw = concordance_index_ipcw(y_train, y_test, -model.predict(X_test), tau=7)[0]
# print(f"LightGBM Survival Model Concordance Index IPCW on train: {train_ci_ipcw:.3f}")
# print(f"LightGBM Survival Model Concordance Index IPCW on test: {test_ci_ipcw:.3f}")
# lightgbm_score_method = f"score_{train_ci_ipcw:.3f}_{test_ci_ipcw:.3f}"

# # Predict and save the results
# if GLOBAL["save_lgbm"]:
#     predict_and_save(X_eval, model, method=f"{size_method}-{lightgbm_score_method}-{clinical_method}-{molecular_method}-{merge_method}-{PARAMS['xgb']['max_depth']}_lr{PARAMS['xgb']['learning_rate']}")


##############################################
# Fit a Random Survival Forest model
##############################################

if GLOBAL["rsf"]["run"]:

    rsf_params_method = "_".join([(str(key) + "=" + str(PARAMS['rsf'][key])) for key in PARAMS['rsf'].keys()])

    rsf = RandomSurvivalForest(**PARAMS["rsf"], random_state=42)
    rsf.fit(X_train, y_train)
    rsf_score_method = score_method(rsf, X_train, X_test, y_train, y_test)

    if GLOBAL["rsf"]["save"]:
        predict_and_save(X_eval, rsf, method=f"{size_method}-{rsf_score_method}-{clinical_method}-{molecular_method}-{merge_method}-{rsf_params_method}")

    if GLOBAL["rsf"]["shap"]:
        from src.report import ShapReport

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)

        report_rsf = ShapReport(model=rsf, X_train=X_train, predict_function=rsf.predict)
        report_rsf.generate_report(output_html=f"report/shap/shap_{rsf.__class__.__name__}_{size_method}-{rsf_score_method}-{clinical_method}-{molecular_method}-{merge_method}-{rsf_params_method}.html")
