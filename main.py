from src.utilities import create_entity, load_target, predict_and_save
from src.preprocess import process_categories, process_missing_values, preprocess_caryotype
import featuretools as ft
import pandas as pd
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw

# clinical = pd.read_csv(f"data/X_train/clinical_train.csv")
# X = clinical["CYTOGENETICS"].apply(preprocess_caryotype)


train = create_entity('train')
X, features_defs = ft.dfs(entityset=train, target_dataframe_name="clinical")
X = process_categories(X, method="dummies")
X = preprocess_caryotype(X)

test = create_entity('test')
X_eval, features_defs = ft.dfs(entityset=test, target_dataframe_name="clinical")
X_eval = process_categories(X_eval, method="del")

target = load_target()

# Create the survival data format
X = X.loc[X.index.isin(target['ID'])]
y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train, X_test, X_eval = process_missing_values(X_train, X_test, X_eval, method="impute", strategy="median")

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

# Predict and save the results
predict_and_save(X_eval, cox, method="featuretools_del_impute_median")