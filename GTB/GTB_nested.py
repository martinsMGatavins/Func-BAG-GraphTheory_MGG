#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:01:18 2022
Gradient tree boosting for brain age prediction using nested CV with 
grid search (5 outer x 10 inner folds)

Grid search on number of estimators and max tree depth

# Argument 1: test data directory
# Argument 2: train data directory
# Argument 3: name of data category
@author: martinsmarksgatavins
"""

#%% Package import
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import shap
import sys
import joblib
import time

#%% Data parsing (make sure data does not have an index column!)
train = pd.read_csv(sys.argv[1]) # full path!
test = pd.read_csv(sys.argv[2]) # full path!
workdir = sys.argv[3] # name & full path of outputdir!

# Splitting off features from full df
X_train = train.drop(columns=["sub","age"],axis=1)
X_test = test.drop(columns=["sub","age"],axis=1).to_numpy()

features = X_train.columns.values

X_train = X_train.to_numpy()

# Splitting off lables from full df
y_train = train['age'].to_numpy().ravel()
y_test = test['age'].to_numpy().ravel()

test_ids = test['sub'].to_numpy().ravel()

#%% Main test_train_split & fold setup
outer_kf = KFold(n_splits=5,shuffle=True,random_state=20)
inner_kf = KFold(n_splits=10,shuffle=True,random_state=20)

#%% Setting up parameter grid (for GridSearchCV)
parameters = {"n_estimators":[25, 50, 75, 100, 200, 300, 400, 500],
               "max_depth":[3, 4, 5, 6, 7, 8]}

# Initializing
gtb = GradientBoostingRegressor()

#%% Training using nested cross-validation + model analysis
print("Training start")
start= time.time()

# Inner loop: 10-fold grid search
modelTrain = GridSearchCV(estimator=gtb,param_grid=parameters,
                          n_jobs=-1,cv=inner_kf)

# Outer loop: 5-fold cross-validation
scores = cross_val_score(modelTrain,X=X_train,y=y_train,
                         cv=outer_kf,n_jobs=-1,verbose=5)

# If n_jobs=-1, this uses all available CPUs to speed up training
with joblib.parallel_backend('multiprocessing'):
    modelTrain.fit(X_train,y_train)
    
# Predicting brain age on test set (using best_estimator)
y_pred = modelTrain.predict(X_test)

# Selecting best model
gtb_best = modelTrain.best_estimator_
params = gtb_best.get_params()

# Error analysis - calculated MSE, MAE, R2 on test set
modelErrors= {'MSE':[mean_squared_error(y_test,y_pred)],
              'MAE':[mean_absolute_error(y_test,y_pred)],
              'R2 score':[gtb_best.score(X_test,y_test)],
              'Num_estimators':[params['n_estimators']],
              'Tree_depth':[params['max_depth']]}

### Feature importance analysis on test data
# Mean decrease in impurity
impurity= gtb_best.feature_importances_
# Permutation importance (10 runs)
pm_importance = permutation_importance(estimator=gtb_best,X=X_test,y=y_test,
                                       n_repeats=10,n_jobs=-1)
# Shapley values: generic & interaction values
explainer = shap.TreeExplainer(gtb_best)
shap_values = explainer.shap_values(X_test,y_test,check_additivity=True) #(n_features) values per subject
mean_shapvalues = np.mean(shap_values,axis=0)
shap_interaction = explainer.shap_interaction_values(X_test) #(n_features)^2 values per subject
mean_interactions = np.mean(shap_interaction,axis=0)

#%% Table with pred & chronological age + performance reports
ba_ca_results = pd.DataFrame({'sub':test_ids,
                              'pred_age':y_pred,
                              'real_age':y_test,
                              'age_gap':y_pred-y_test})

# saves age results table to workdir
ba_ca_results.to_csv(workdir + "/brainages.csv",index=False)


modelErrors= {'MSE':[mean_squared_error(y_test,y_pred)],
              'MAE':[mean_absolute_error(y_test,y_pred)],
              'R2 score':[gtb_best.score(X_test,y_test)],
              'Num_estimators':[params['n_estimators']],
              'Tree_depth':[params['max_depth']]}

# saves model error table to workdir
pd.DataFrame(modelErrors).to_csv(workdir + "/GTB_stats.csv",index=False)

feature_importances = {'features':features,
                        'MDI':impurity,
                        'Mean PI':pm_importance.importances_mean,
                        'PI standard dev':pm_importance.importances_std,
                        'Mean Shapley values':mean_shapvalues}

# saves importance & interaction table to workdir
pd.DataFrame(feature_importances).to_csv(workdir + "/FeatureImportanes.csv",index=False)
pd.DataFrame(mean_interactions).to_csv(workdir + "/FeatureInteractions.csv",index=False)

# Save best-fit GTB model as an .sav file (use joblib.load to re-use)
joblib.dump(gtb_best,workdir + "/GTB_brainage.sav")
print("Ending time in minutes:",(time.time()-start)/60.0)
        
