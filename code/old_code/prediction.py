#pip install julearn

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from pingouin import intraclass_corr
from multiprocessing import pool
from scipy.stats import pearsonr
import shap
from julearn.model_selection import ContinuousStratifiedKFold

# Paths
DATA_DIR = "../../data"
OUTPUT_DIR = "../../predictions"

# Variables
global OUTER_FOLDS
global INNER_FOLDS
global OUTER_SEED
global INNER_SEED

OUTER_FOLDS = 5
INNER_FOLDS = 10
OUTER_SEED = 300
INNER_SEED = 500

# Functions
# Preprocesses the data by: 
# including covariates in the feature space (concatenate);
# residualizing covariates from each feature.
def preprocess_data(X_train, X_test, covariates_train, covariates_test):
    X_concat_train = np.hstack((X_train, covariates_train))
    X_concat_test = np.hstack((X_test, covariates_test))
    X_processed_train = X_train
    X_processed_test = X_test
    for feature in X_train:
        model = LinearRegression()
        model.fit(covariates_train, X_train[feature])
        X_train_pred = model.predict(covariates_train)
        residuals_train = X_train[feature] - X_train_pred
        std_residuals_train = residuals_train / np.std(residuals_train, axis=0)  # Standardize residuals
        X_processed_train[feature] = std_residuals_train

        X_test_pred = model.predict(covariates_test)
        residuals_test = X_test[feature] - X_test_pred
        std_residuals_test = residuals_test / np.std(residuals_test, axis=0)  # Standardize residuals
        X_processed_test[feature] = std_residuals_test
    
    return X_concat_train, X_concat_test, X_processed_train, X_processed_test

# Used just for the GTB models
def train_grid(model,paramgrid,X,y,param=None):
    
    rkf_inner = KFold(X,y,n_splits=INNER_FOLDS)
    if param == None:
        gcv = GridSearchCV(model,paramgrid,cv=rkf_inner,refit=True,n_jobs=-1)
    else:
        gcv = GridSearchCV(model,paramgrid,cv=rkf_inner,refit=param,n_jobs=-1)

    gcv.fit(X,y)

    return gcv.best_estimator_

# Unfinished
def train_grid_haufe_transform(X,y,inner_kf):
    
    env = ElasticNetCV(cv=inner_kf)
    env.fit(X,y)
    
    # Applying Haufe trick [Haufe 2014]
    coef_raw = env.coef_
    coef_trans = np.cov(X.T).dot(coef_raw.T.dot(inv_Y)).T

    return coef_raw, coef_trans, env

# Unfinished
def train_and_evaluate_fold_outer(X_train, y_train, covariates_train, train_index, test_index, config, i):
    Xk_train, Xk_test = X_train[train_index], X_train[test_index]
    yk_train, yk_test = y_train[train_index], y_train[test_index]
    covariates_k_train, covariates_k_test = covariates_train[train_index], covariates_train[test_index]
    
    _, Xk_test_processed = preprocess_data(Xk_train, Xk_test, yk_train, yk_test, covariates_k_train, covariates_k_test, config)
    
    rkf_inner = KFold(Xk_train,yk_train,n_splits=INNER_FOLDS,random_state=INNER_SEED + i)
    best_gtb_models = []
    gtb_gain_folds = []
    best_elasticnet_models = []
    elasticnet_fold_weights = [] 
    return best_gtb_models, gtb_gain_folds, best_elasticnet_models, elasticnet_fold_weights

# Unfinished
def permutation_importance_scores(model,X,y):
    scores = permutation_importance(model,X,y)
    return scores

# Not tested
def icc(weights):
    weights_melted = pd.melt(weights,id_vars=['feature'])
    icc = []
    for feature in weights['feature']:
        full_icc = intraclass_corr(weights_melted,targets=feature,raters='variable',ratings='value')
        icc_3_1 = full_icc.loc[full_icc.get_loc('ICC3'),'ICC']
        icc_3_1_ci = full_icc.loc[full_icc.get_loc('ICC3'),'95%CI']

        icc_3_k = full_icc.loc[full_icc.get_loc('ICC3k'),'ICC']
        icc_3_k_ci = full_icc.loc[full_icc.get_loc('ICC3k'),'95%CI']
        icc.hstack([icc_3_1,icc_3_1_ci,icc_3_k,icc_3_k_ci])
    return icc

def simple_report(model,y_real):
# UNFINISHED (should create and save a CSV file)
    return

# Main code
if len(sys.argv) != 4:
    print("Usage: python3 code.py feature labels sens")
    sys.exit(1)

# Read in the arguments
split_vector_file = sys.argv[1]
feature = sys.argv[2]
label = sys.argv[3]
sensitivity = sys.argv[4].lower() == 'true'

# Read data
split_vector = np.loadtxt(split_vector_file)
features_df = pd.read_csv(features_file, index_col="subject_id")
y = pd.read_csv(labels_file, index_col="subject_id")
covariates = features_df["mean_FD"]
X = features_df.drop(columns=["sub","mean_FD"])

# Test/validation split using anticlustered indices
X_train, X_val = X[split_vector==1], X[split_vector==2]
y_train, y_val = y[split_vector==1], y[split_vector==2]
covariates_train, covariates_val = covariates[split_vector==1], covariates[split_vector==2]

# Continous decile-stratified K-fold split (from julearn)
rkf_outer = ContinuousStratifiedKFold(n_bins=OUTER_FOLDS,methods="quantile",n_splits=OUTER_FOLDS,random_state=OUTER_SEED)

# Confound regression from validaton set
_, X_val_resid, _, X_val_concat = preprocess_data(X_train, X_val, y_train, y_val, covariates_train, covariates_val, 'with_covariates')

# Initializing model storage
gtb_keys = ['models','models_concat','MAE','gain','permutations','shapscores']
gtb_models = dict(zip(gtb_keys, []*len(gtb_keys)))

en_keys = ['models','R2','intercept','raw_weights','haufe_weights','shapscores']
en_models = dict(zip(en_keys, []*len(en_keys)))

# Run folds: 
for train_index, test_index in rkf_outer.split(X_train,y_train):
    Xk_train, Xk_test = X_train[train_index], X_train[test_index]
    yk_train, yk_test = y_train[train_index], y_train[test_index]
    covariates_k_train, covariates_k_test = covariates_train[train_index], covariates_train[test_index]
    
    # For each fold, the training folds are used as the train set for the residualizing regressor
    # the test folds are residualized used the trained residualizing regressor (prevents data "leakage")
    Xk_train_concat, Xk_test_concat, Xk_train_regressed, Xk_test_regressed = preprocess_data(Xk_train, Xk_test, yk_train, yk_test, covariates_k_train, covariates_k_test)
    
    # Inner CV for hyperparameter tuning and model fitting
    inner_kf = KFold(Xk_train,yk_train,n_splits=5)

    GTB_params = {

    }
    
    gtb = xgb.XGBRegressor()
    gtb_model = train_grid(gtb,GTB_params,inner_kf,Xk_train_regressed,yk_train)
    gtb_models['models'].append(gtb_model)

    gtb_model_nonreg = train_grid(gtb,GTB_params,inner_kf,Xk_train_concat,yk_train)
    gtb_models['models_concat'].append(gtb_model_nonreg)

    gtb_gain = gtb_model.get_score(importance_type='gain')
    gtb_models['gain'].vstack(gtb_gain)

    gtb_permutation = permutation_importance(gtb_model,Xk_test_regressed,yk_test)
    gtb_models['permutations'].vstack(gtb_permutation)

    explainer = shap.explainer(gtb_model)
    gtb_shap = explainer.shap_values(Xk_test_regressed)
    gtb_models['shapscores'].vstack(gtb_shap)

    y_preds = gtb_model.fit(Xk_train_regressed)
    gtb_models['MAE'].append(mean_absolute_error(y_preds,y_train))

    coef_raw, coef_trans, en = train_grid_haufe_transform(Xk_train_concat,yk_test,inner_kf)
    en_models['models'].append(en)
    en_models['intercept'].append(en.intercept_)

    en_models['raw_weights'].vstack(coef_raw)
    en_models['haufe_weights'].vstack(coef_trans)

    elasticnet_explainer = shap.explainer(en)
    en_models['shapscores'].vstack(elasticnet_explainer.shap_values(Xk_test_concat))

    y_en_preds = en.predict(Xk_test_concat)
    en_models['R2'].append(en.score(Xk_test_concat,yk_test))


best_gtb_idx = gtb_models['MAE'].index(min(gtb_models['MAE']))
gtb_icc_gain = icc(gtb_models['gain',best_gtb_idx])
gtb_icc_permutation = icc(gtb_models['permutation',best_gtb_idx])
gtb_icc_shap = icc(gtb_models['shapscores',best_gtb_idx])
best_gtb = gtb_models['MAE',best_gtb_idx]
best_gtb_preds = best_gtb.predict(X_val_resid)
simple_report(best_gtb_preds,y_val)

best_elasticnet_idx = en_models['R2'].index(min(en_models['R2']))
en_icc_raw = icc(en_models['raw_weights',best_elasticnet_idx])
en_icc_haufe = icc(en_models['haufe_weights',best_elasticnet_idx])
en_icc_shap = icc(en_models['shapscores',best_elasticnet_idx])
best_elasticnet_model = en_models['models',best_elasticnet_idx]
elasticnet_preds = best_elasticnet_model.predict(X_val_concat)
simple_report(elasticnet_preds,y_val)

elasticnet_avg = LinearRegression()
elasticnet_avg.intercept_ = np.mean(en_models['intercept'],0)
elasticnet_avg.coef_ = np.mean(en_models['raw_weights'],0)
elasticnet_avg_preds = elasticnet_avg.predict(X_val_concat)
simple_report(elasticnet_avg_preds,y_val)

elasticnet_avg_haufe = LinearRegression()
elasticnet_avg_haufe.intercept_ = np.mean(en_models['intercept'],0)
elasticnet_avg_haufe.coef_ = np.mean(en_models['haufe_weights'],0)
elasticnet_avg_haufe_preds = elasticnet_avg_haufe.predict(X_val_concat)
simple_report(elasticnet_avg_haufe_preds,y_val)
