from src.utilities import create_entity, predict_and_save, split_data
from src.preprocess import process_missing_values, main_preprocess
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import lightgbm as lgb

import warnings
import logging

# Régler le logger de Featuretools au niveau ERROR
logging.getLogger('featuretools.entityset').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",message=".*Ill-conditioned matrix.*")

data = create_entity()

# Specify the columns to be processed
save = True

training_size = 0.7
clinical_process = ["CYTOGENETICS"] # Possible: ["CYTOGENETICS", "HB/PLT", "logMONOCYTES", "logWBC", "logANC"]
molecular_process = []#["GENE", "END-START"] # Possible: ["GENE", "EFFECT", "ALT", "REF", "END-START"]
merge_process = ["gpt"] # Possible: ["featuretools"]

data = main_preprocess(data, clinical_process, molecular_process, merge_process)
X, X_eval, y = split_data(data)
# Check if there are any columns that are not float or int in X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - training_size), random_state=42)

X_train, X_test, X_eval = process_missing_values(X_train, X_test, X_eval, method="impute", strategy="median")
# Ensure all columns in X_train are either float or int

##############################################
# Define the methods used for training
##############################################

size_method = f"size_{training_size}"
clinical_method = "clinical_" + "_".join(clinical_process).replace("/", "divby")
molecular_method = "molecular_" + "_".join(molecular_process).replace("/", "divby")
merge_method = "merge_" + "_".join(merge_process).replace("/", "divby")

##############################################
# Fit a CoxPH model
##############################################

# Initialize and train the Cox Proportional Hazards model
cox = CoxPHSurvivalAnalysis()
cox.fit(X_train, y_train)

# Evaluate the model using Concordance Index IPCW
cox_cindex_train = concordance_index_ipcw(y_train, y_train, cox.predict(X_train), tau=7)[0]
cox_cindex_test = concordance_index_ipcw(y_train, y_test, cox.predict(X_test), tau=7)[0]
print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.2f}")
print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.2f}")
cox_score_method = f"score_{cox_cindex_train:.3f}_{cox_cindex_test:.3f}"

# Predict and save the results
if save:
    predict_and_save(X_eval, cox, method=f"{size_method}-{cox_score_method}-{clinical_method}-{molecular_method}-{merge_method}")


##############################################
# Fit a Gradient Boosting model
##############################################

xgb_params = {
    'max_depth': 2,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.5,
}

xgb_params_method = "_".join([(str(key) + "=" + str(xgb_params[key])) for key in xgb_params.keys()])

xgb = GradientBoostingSurvivalAnalysis(subsample=xgb_params['subsample'], n_estimators=xgb_params['n_estimators'], learning_rate=xgb_params['learning_rate'], max_depth=xgb_params['max_depth'], random_state=42)
xgb.fit(X_train, y_train)
xgb_cindex_train = concordance_index_ipcw(y_train, y_train, xgb.predict(X_train), tau=7)[0]
xgb_cindex_test = concordance_index_ipcw(y_train, y_test, xgb.predict(X_test), tau=7)[0]
print(f"Gradient Boosting Survival Model Concordance Index IPCW on train: {xgb_cindex_train:.2f}")
print(f"Gradient Boosting Survival Model Concordance Index IPCW on test: {xgb_cindex_test:.2f}")
xgboost_score_method = f"score_{xgb_cindex_train:.3f}_{xgb_cindex_test:.3f}"

if save:
    predict_and_save(X_eval, xgb, method=f"{size_method}-{xgboost_score_method}-{clinical_method}-{molecular_method}-{merge_method}-{xgb_params_method}")


##############################################
# Fit a LightGBM model
##############################################

# Define LightGBM parameters
lgbm_params = {
    'max_depth': 2,
    'learning_rate': 0.05,
    'verbose': 0
}

X_train_lgb = X_train  # Features for training
y_train_transformed = y_train['time']

# Create LightGBM dataset
train_dataset = lgb.Dataset(X_train_lgb, label=y_train_transformed)

# Train the LightGBM model
model = lgb.train(params=lgbm_params, train_set=train_dataset)

# Evaluate the model using Concordance Index IPCW
train_ci_ipcw = concordance_index_ipcw(y_train, y_train, -model.predict(X_train), tau=7)[0]
test_ci_ipcw = concordance_index_ipcw(y_train, y_test, -model.predict(X_test), tau=7)[0]
print(f"LightGBM Survival Model Concordance Index IPCW on train: {train_ci_ipcw:.2f}")
print(f"LightGBM Survival Model Concordance Index IPCW on test: {test_ci_ipcw:.2f}")
lightgbm_score_method = f"score_{train_ci_ipcw:.3f}_{test_ci_ipcw:.3f}"

# Predict and save the results
if save:
    predict_and_save(X_eval, model, method=f"{size_method}-{lightgbm_score_method}-{clinical_method}-{molecular_method}-{merge_method}-{lgbm_params['max_depth']}_lr{lgbm_params['learning_rate']}")