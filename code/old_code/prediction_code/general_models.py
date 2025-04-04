import sys
import numpy as np
import pandas as pd
import sklearn.linear_model
import xgboost as xgb
import sklearn
import joblib
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from pingouin import intraclass_corr
from multiprocessing import pool
from scipy.stats import pearsonr
import shap
from julearn.model_selection import ContinuousStratifiedKFold
import os

# Paths
DATA_DIR = "../data"
OUTPUT_DIR = "../predictions"

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
def residualize(X, covariates, train_idx, test_idx):
    X_train, X_test = X[train_idx], X[test_idx]
    covariates_train, covariates_test = covariates[train_idx], covariates[test_idx]
    X_train_resid, X_test_resid = X_train, X_test

    for feature in X_train:
        model = LinearRegression()
        model.fit(covariates_train, X_train[feature])
        X_train_pred = model.predict(covariates_train)
        residuals_train = X_train[feature] - X_train_pred
        std_residuals_train = residuals_train / np.std(residuals_train, axis=0)  # Standardize residuals
        X_train_resid[feature] = std_residuals_train

        X_test_pred = model.predict(covariates_test)
        residuals_test = X_test[feature] - X_test_pred
        std_residuals_test = residuals_test / np.std(residuals_test, axis=0)  # Standardize residuals
        X_test_resid[feature] = std_residuals_test
    
    return X_train_resid, X_test_resid 

def concatenate(X, covariates, train_idx, test_idx):
    X_train, X_test = X[train_idx], X[test_idx]
    covariates_train, covariates_test = covariates[train_idx], covariates[test_idx]
    X_train_concat = np.hstack((X_train, covariates_train))
    X_test_concat = np.hstack((X_test, covariates_test))

    return X_train_concat, X_test_concat

# Supports all models
def grid_search(model,X,y,paramgrid,n_folds,seed,param="MAE",n_workers=-1):
    kf = ContinuousStratifiedKFold(X,y,n_splits=n_folds,random_state=seed)
    gcv = GridSearchCV(model,paramgrid,cv=kf,refit=param,n_jobs=n_workers)
    gcv.fit(X,y)

    return gcv.best_estimator_

# Supports only linear regressors
def linreg_grid_search(X,y,n_folds,seed,model="ElasticNet",n_workers=-1):
    kf = ContinuousStratifiedKFold(X,y,n_splits=n_folds,random_state=seed)
    if model == "ElasticNet":
        regressor = ElasticNetCV(cv=kf,n_jobs=n_workers)
    elif model == "Lasso":
        regressor = LassoCV(cv=kf,n_jobs=n_workers)
    else:
        raise ValueError("Incorrect model type entered, choose 'ElasticNet' or 'Lasso'")
    regressor.fit(X,y)

    return regressor

# Haufe transform - implement!
def haufe(X,y,coef):
    pass

def model_eval(model,X,y):
    y_pred = model(X)
    MAE = mean_absolute_error(y_pred=y_pred,y_true=y)
    MAPE = mean_absolute_percentage_error(y_pred=y_pred,y_true=y)

    if isinstance(model,sklearn.linear_model):
        R2 = model.score(X,y)
    else:
        R2 = r2_score(y_pred=y_pred,y_true=y)

    corr = pearsonr(y_pred,y)
    return MAE, MAPE, R2, corr

# 
def reliability(model_list,X,y,features):
    weights = [features]
    for model in model_list:
        model_FI = feature_importance_all(model,X,y)
        weights.hstack(model_FI.get("importance"))
    weights_long = pd.wide_to_long(pd.DataFrame(weights))
    
    # wide to long transform
    icc_full = intraclass_corr(weights_long,targets="Variables",raters="Model",ratings="Weights")
    icc = [icc_full["Type"],icc_full["ICC"],icc_full["pval"]]

    return icc

def feature_importance_all(model,X=None,y=None,permutation_importance=False,intercepts=False,shapley=False,X_train=None):
    feature_importances = dict()
    if isinstance(model,xgb.XGBRegressor):
        feature_importances["importance"] = model.feature_importance(type="gain")
    elif isinstance(model,sklearn.linear_model):
        feature_importances["importance"] = model.coef_
        if intercepts:
            feature_importances["intercept"] = model.intercept_
    else:
        raise ValueError("Incorrect model type entered, function supports GTB and linear regressors only")
    
    if permutation_importance:
        perm_importance_mean,perm_importance_sd,_ = permutation_importance(model,X,y)
        feature_importances["perm_imp_mean"] = perm_importance_mean
        feature_importances["perm_imp_sd"] = perm_importance_sd
    
    if shapley:
        if isinstance(model,sklearn.linear_model):
            explainer = shap.Explainer(model.predict,X_train)
            shap_vals = explainer(X)
        elif isinstance(model,xgb.XGBRegressor):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer(X)
        feature_importances["shap_mean"] = np.mean(shap_vals)
        feature_importances["shap_distribution"] = [X,shap_vals]

    return feature_importances

def model_averaging(model_list):
    coefficients = []
    intercepts = []
    for model in model_list:
        coefficients = coefficients.hstack(model.coef_)
        intercepts = intercepts.append(model.intercept_)
    avg_coef = np.mean(coefficients,axis=1) # check this
    avg_intercept = np.mean(intercepts)
    averaged_model = LinearRegression()
    averaged_model.coef_ = avg_coef
    averaged_model.intercept_ = avg_intercept
    return averaged_model

def outer_fold(data):
    X, y, covars, train_idx, test_idx, randomval = data
    X_train, X_val = residualize(X,covars,train_idx,test_idx)
    y_train, y_val = y[train_idx],y[test_idx]
    paramgrid = {} # complete grid
    gtb_trained = grid_search(xgb.XGBRegressor(),
                              paramgrid,
                              X_train,y_train,
                              n_folds=INNER_FOLDS,
                              seed=randomval,
                              n_workers=INNER_FOLDS)
    GTB_metrics = model_eval(gtb_trained,X_val,y_val)
    #GTB_FIs = feature_importance_all(gtb_trained)


    X_train, X_val = concatenate(X,covars,train_idx,test_idx)
    elasticNet_trained = linreg_grid_search(X_train,
                                    y_train,
                                    n_folds=INNER_FOLDS,
                                    n_workers=INNER_FOLDS)
    LinRegEN_metrics = model_eval(elasticNet_trained,X_val,y_val)
    #LinReg_FIs = feature_importance_all(elasticNet_trained)

    return GTB_metrics, gtb_trained, LinRegEN_metrics, elasticNet_trained
    
def evaluate(model,X,y,mae=True,mape=True,coeff_of_determination=True,correlation=True):
    y_pred = model.predict(X)
    error = y - y_pred
    metrics = dict()
    if mae:
        mae_val = np.mean(error)
        metrics["mae"] = mae_val
    if mape:
        mape_val = mean_absolute_percentage_error(y,y_pred)
        metrics["mape"] = mape_val
    if coeff_of_determination:
        if isinstance(model,sklearn.linear_model):
            r2 = model.score(X)
        else:
            r2 = r2_score(y,y_pred)
        metrics["r2"] = r2
    if correlation:
        corr = pearsonr(y,y_pred)
        metrics["corr"] = corr

    return error, metrics
    
if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        raise ValueError("Usage: python3 code.py variable split_vector.csv features.csv labels.csv")
    os.chdir(DATA_DIR)

    # Read in the arguments
    variable_name = sys.argv[1]
    split_vector_file = sys.argv[2]
    features_file = sys.argv[3]
    labels_file = sys.argv[4]

    # Read data
    split_vector = np.loadtxt(split_vector_file)
    features_df = pd.read_csv(features_file, index_col="subject_id")
    y = pd.read_csv(labels_file, index_col="subject_id")
    covariates = features_df["mean_FD"]
    X = features_df.drop(columns=["sub","mean_FD"])
    train_idx, val_idx = split_vector[split_vector==1], split_vector[split_vector==2]

    X_train, y_train = X[train_idx], y[train_idx]

    kf = ContinuousStratifiedKFold(X_train,y_train,n_splits=OUTER_FOLDS,random_state=OUTER_SEED)
    folds = [(X_train,y_train,covariates,i,j,INNER_SEED) for i,j in kf.split(X_train,y_train)]

    # Cross-validation (5 outer, 10 nested, inner loops)
    results = joblib.Parallel(n_jobs=OUTER_FOLDS*INNER_FOLDS)(
        joblib.delayed(outer_fold)(fold) for fold in folds
    )

    gtb_metrics, gtb_models, en_metrics, en_models = np.array(results).T

    _, X_val_resid = residualize(X,covariates,train_idx=train_idx,test_idx=val_idx)
    _, X_val_concat = concatenate(X,covariates,train_idx=train_idx,test_idx=val_idx)
    y_val = y[val_idx]

    # Model selection from outer folds
    best_gtb = gtb_models[min(gtb_metrics)]
    best_elastic = en_models[min(en_models)]
    average_elastic = model_averaging(en_models)

    # Calculate errors and model performance metrics (MAE, MAPE, R2, r)
    best_gtb_error, best_gtb_metrics = evaluate(best_gtb,X=X_val_resid,y=y_val)
    best_elastic_error, best_elastic_metrics = evaluate(best_elastic,X=X_val_concat,y=y_val)
    avg_elastic_error, avg_elastic_metrics = evaluate(average_elastic,X=X_val_concat,y=y_val)

    # Feature importances
    best_gtb_importance = feature_importance_all(best_gtb,X_val_resid,y_val,
                                                 permutation_importance=True,
                                                 intercepts=False,shapley=True)
    best_elastic_importance = feature_importance_all(best_elastic,X_val_concat,y_val,
                                                     permutation_importance=True,
                                                     intercepts=False,shapley=True)
    avg_elastic_importance = feature_importance_all(average_elastic,X_val_concat,y_val,
                                                     permutation_importance=True,
                                                     intercepts=False,shapley=True)

    # Feature importance reliabilities
    gtb_reliability = reliability(gtb_models,X_val_resid,y_val)
    elastic_net_reliability = reliability(gtb_models,X_val_concat,y_val)
    elastic_haufe_reliability = reliability(haufe(gtb_models),X_val_concat,y_val)

    os.chdir(OUTPUT_DIR + variable_name)
    
    # Save all of the csvs
    errors = pd.DataFrame(gtb_error=best_gtb_error,
                          en_error=best_elastic_error,
                          avg_en_error=avg_elastic_error) 
    errors.to_csv(variable_name + "_metrics.csv",index=False)

    gtb_en_long = pd.concat(gtb_reliability,elastic_net_reliability,axis=0,ignore_index=True)
    reliability_long = pd.concat(gtb_en_long,elastic_haufe_reliability,axis=0,ignore_index=True)
    reliability_wide = reliability_long.pivot(index="Id",columns="Type",Values="ICC")
    reliability_wide.to_csv(variable_name + "_ICC.csv",index=False)

    gtb_metrics["model"] = "GTB"
    best_elastic_metrics["model"] = "best_EN"
    avg_elastic_metrics["model"] = "avg_EN"
    gtb_metrics = pd.DataFrame.from_dict(best_elastic_metrics,orient="index")
    best_en_metrics = pd.DataFrame.from_dict(best_gtb_metrics,orient="index")
    avg_en_metrics = pd.DataFrame.from_dict(avg_elastic_metrics,orient="index")
    gtb_en = pd.concat(gtb_metrics,best_en_metrics,axis=0,ignore_index=True)
    metrics = pd.concat(gtb_en,avg_en_metrics,axis=0,ignore_index=True)
    metrics.to_csv(variable_name + "_model_performance.csv",index=False)

    best_gtb_shap = pd.DataFrame(best_gtb_importance.pop("shap_distribution"))
    best_gtb_shap.to_csv(variable_name + "_GTB_shap_values.csv",index=False)
    
    best_gtb_importance_df = pd.DataFrame.from_dict(best_gtb_importance,orient="index")
    best_gtb_importance_df.to_csv(variable_name + "_GTB_importances.csv",index=False)

    best_en_shap = pd.DataFrame(best_elastic_importance.pop("shap_distribution"))
    best_en_shap.to_csv(variable_name + "_bestEN_shap_values.csv",index=False)

    best_en_importance_df = pd.DataFrame.from_dict(best_elastic_importance,orient="index")
    best_en_importance_df.to_csv(variable_name + "_bestEN_importances.csv",index=False)

    avg_en_shap = pd.DataFrame(avg_elastic_importance.pop("shap_distribution"))
    avg_en_shap.to_csv(variable_name + "_avgEN_shap_values.csv",index=False)

    avg_en_shap_df = pd.DataFrame.from_dict(avg_elastic_importance,orient="index")
    avg_en_shap_df.to_csv(variable_name + "_avgEN_importances.csv",index=False)
    
    # write each of the 6 tables (3 importances, 3 SHAP distributions)

    

