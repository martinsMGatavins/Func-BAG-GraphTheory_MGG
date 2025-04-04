import sys
import numpy as np
import pandas as pd
import sklearn.linear_model
import xgboost as xgb
import sklearn
import joblib
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV, LassoCV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from pingouin import intraclass_corr
from multiprocessing import pool
from scipy.stats import pearsonr
import shap
# from julearn.model_selection import ContinuousStratifiedKFold
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

def crossval_run(fold, data):
    X, y, covars, train_idx, test_idx = data
    X_train, X_test = residualize(X,covars,train_idx,test_idx)
    y_train, y_test = y[train_idx],y[test_idx]
    gtb_param_space = {
        'n_estimators':[5, 15, 25, 35, 50, 75, 100, 150, 200, 250, 300, 400, 550, 750, 1000],
        'learning_rate':[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0],
        'max_depth':[3]
    }

    # 10-fold CV
    kf = KFold(X,y,n_splits=INNER_FOLDS,random_state=INNER_SEED)

    # GridSearch for best parameter set
    gcv = GridSearchCV(xgb.XGBRegressor(),param_grid=gtb_param_space,cv=kf,n_jobs=INNER_FOLDS)
    gcv.fit(X, y, refit=True)
    fitted_gtb = gcv.best_estimator_
    y_gtb = fitted_gtb.predict(X_test)

    X_train, _ = concatenate(X,covars,train_idx,test_idx)
    ecv = ElasticNetCV(cv=kf,n_jobs=INNER_FOLDS)
    ecv.fit(X_train,y_train,refit=True)
    y_elasticnet = ecv.predict(X_test)

    output = pd.DataFrame({"fold":fold,
                           "y_actual":y_test,
                           "y_gtb":y_gtb,
                           "y_elastic":y_elasticnet})
    gtb_params = pd.DataFrame(gcv.best_params_)
    en_params = pd.DataFrame({"L1":ecv.l1_ratio_,
                              "alpha":ecv.alpha_})

    return output, gtb_params, en_params, fitted_gtb, ecv

# def cross_val_run(data):
#     X, y, covars, kf, gtb_params, en_params = data
#     model_gtb_mae_fold = []
#     model_en_negmse_fold = []
#     trained_gtb_models = []
#     trained_en_models = []
#     gtb = xgb.XGBRegressor(n_estimators = gtb_params['n_estimators'],
#                            max_depth = 3,
#                            learning_rate = gtb_params["learning_rate"])
#     en = ElasticNet(alpha = en_params["alpha"],
#                     l1_ratio = en_params["l1_ratio"])
#     for train_idx, test_idx in kf.split(X,y):
#         X_train, X_test = residualize(X,covars,train_idx,test_idx)
#         y_train, y_test = y[train_idx],y[test_idx]
#         fit_model = gtb.fit(X_train,y_train)
#         trained_gtb_models.append(fit_model)
#         model_gtb_mae_fold.append(mean_absolute_error(y_test,fit_model.predict(X_test)))

#         X_train, X_test = concatenate(X,covars,train_idx,test_idx)
#         en.fit(X_train,y_train)
#         model_en_negmse_fold.append(-1*mean_squared_error(y_test,en.predict(X_test)))

#     best_gtb = trained_gtb_models[min(range(len(model_gtb_mae_fold)), key=model_gtb_mae_fold.__getitem__)]
#     best_en = trained_en_models[max(range(len(model_en_negmse_fold)), key=model_en_negmse_fold.__getitem__)]

#     return best_gtb, best_en
    
if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        raise ValueError("Usage: python3 code.py variable label split_vector.csv features.csv labels.csv")
    os.chdir(DATA_DIR)

    # Read in the arguments
    variable_name = sys.argv[1]
    label_name = sys.argv[2]
    split_vector_file = sys.argv[3]
    features_file = sys.argv[4]
    labels_file = sys.argv[5]

    # Read data
    split_vector = np.loadtxt(split_vector_file)
    features_df = pd.read_csv(features_file, index_col="subject_id")
    y = pd.read_csv(labels_file, index_col="subject_id")
    covariates = features_df["mean_FD"]
    X = features_df.drop(columns=["sub","mean_FD"])
    train_idx, val_idx = split_vector[split_vector==1], split_vector[split_vector==2]

    X_train, y_train, covariates_train = X[train_idx], y[train_idx], covariates[train_idx]

    # 5-fold CV (outer folds)
    kf = KFold(X_train,y_train,n_splits=OUTER_FOLDS,random_state=OUTER_SEED)
    folds = [(X_train,y_train,covariates,i,j) for i,j in kf.split(X_train,y_train)]

    # Grid-search Cross-validation (10-fold CV for each of the 5-fold CV folds)
    cross_val_outputs = joblib.Parallel(n_jobs=OUTER_FOLDS*INNER_FOLDS)(
        joblib.delayed(crossval_run)(i, fold) for i, fold in enumerate(folds)
    )
    
    predictions = pd.concat([res[0] for res in cross_val_outputs], ignore_index=True)
    pred, gtb_params, en_params = (pd.concat(dfs, ignore_index=True) for dfs in zip(*cross_val_outputs[:3]))
    gtb_list, en_list = zip(*cross_val_outputs[3:])
    # outer_folds = [(X_train,y_train,covariates,kf,gtb_params,en_params) for gtb_params, en_params in best_parameters]
    # # CV of 5 different models from each GridSearchCV (5-fold CV)
    # best_models = joblib.Parallel(n_jobs=OUTER_FOLDS*OUTER_FOLDS)(
    #     joblib.delayed(cross_val_run)(fold) for fold in outer_folds
    # )

    X_train_resid, X_val_resid = residualize(X,covariates,train_idx=train_idx,test_idx=val_idx)
    X_train_concat, X_val_concat = concatenate(X,covariates,train_idx=train_idx,test_idx=val_idx)
    y_val = y[val_idx]

    models_train_df = pd.DataFrame()
    models_train_df["y_train"] = y_train
    models_val_df = pd.DataFrame()
    models_val_df["y_test"] = y_val
    for i, gtb, en in enumerate(best_models):
        models_train_df[f'GTB_{i+1}'] = gtb.predict(X_train_resid)
        models_train_df[f'EN_{i+1}'] = en.predict(X_train_concat)

        models_val_df[f'GTB_{i+1}'] = gtb.predict(X_val_resid)
        models_val_df[f'EN_{i+1}'] = en.predict(X_val_concat)
    
    gtb_model_list, en_model_list = zip(*best_models)
    
    os.chdir(OUTPUT_DIR + variable_name)
    
    joblib.dump(gtb_model_list,f"{variable_name}_{label_name}_gtb.pkl")
    joblib.dump(en_model_list,f"{variable_name}_{label_name}_elasticnet.pkl")
    
    models_train_df.to_csv(path=f"{variable_name}_{label_name}_train.csv",index=False)
    models_val_df.to_csv(path=f"{variable_name}_{label_name}_test.csv",index=False)

    

