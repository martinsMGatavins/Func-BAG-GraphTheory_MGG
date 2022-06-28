#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:01:18 2022
Gradient tree boosting for brain age prediction using nested CV with 
grid search (10 x 10 folds)

@author: martinsmarksgatavins
"""
# Argument 1: name of output directory ("outputs/${outputdir}")
# Argument 2: name of input csv file ("data/${csv filename}")

#%% Package import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import shap
import sys
import joblib

#%% Data import & directory setup
outputdir = sys.argv[1]
data = pd.read_csv(sys.argv[2])
varnames = data.drop(columns=["sub","age"],axis=1).columns.values
labels = data[["sub","age"]]
X = data.drop(columns=["sub","age"],axis=1).to_numpy()
y = labels.drop(columns=["sub"],axis=1).to_numpy().ravel() # CHANGE HERE?

#%% Main test_train_split (picking a random_state) & setting up CV
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=21)
# get test_split indices (1 - indices OR 2 - have y be the (N,2) with subject + id)

outer_kf = KFold(n_splits=7,shuffle=True,random_state=21)
inner_kf = KFold(n_splits=7,shuffle=True,random_state=21)

#%% Setting up parameters to test (incomplete) for GridSearchCV
parameters = {"n_estimators":[10,20,50],
              "max_depth":[10,20,50],
              "min_samples_split":[2,4,6],
              "n_iter_no_change":[None,3,5]}
gtb = GradientBoostingRegressor()

#%% Training using nested cross-validation + model analysis

# Inner loop: 7-fold grid search
modelTrain = GridSearchCV(estimator=gtb,param_grid=parameters,
                          n_jobs=-1,cv=inner_kf)
# Outer loop: 7-fold cross-validation
scores = cross_val_score(modelTrain,X=X_train,
                         y=y_train,cv=outer_kf)

modelTrain.fit(X_train,y_train)
y_pred = modelTrain.predict(X_test)

gtb_best = modelTrain.best_estimator_
# Error analysis
modelErrors= {'MSE':[mean_squared_error(y_test,y_pred)],
              'MAE':[mean_absolute_error(y_test,y_pred)],
              'R2 score':[gtb_best.score(X_test,y_test)]}

# Feature importance analysis (on test)
impurity= gtb_best.feature_importances_
pm_importance = permutation_importance(estimator=gtb_best,X=X_test,y=y_test,
                                       n_repeats=10,n_jobs=-1,random_state=21)
explainer = shap.TreeExplainer(gtb_best)
shap_values = explainer.shap_values(X_test,y_test,check_additivity=True)
mean_shapvalues = np.mean(shap_values,axis=0)

shap_interaction = explainer.shap_interaction_values(X_test)
mean_interactions = np.mean(shap_interaction,axis=0)

#%% Table with pred & chronological age + performance reports
results = [y_pred, y_test] 
ba_ca_results = pd.DataFrame(results,columns=["pred_age","real_age"])
ba_ca_results.to_csv(outputdir + "/GTB_brainage_results.csv")

# MSE, MAE, R2
pd.DataFrame(modelErrors).to_csv(outputdir + "/GTB_brainage_modelstats.csv")

# MDI, PI, SHAP vals & interactions
feature_importances = {'features':varnames,
                        'MDI':impurity,
                        'Mean PI':pm_importance.importances_mean,
                        'PI standard dev':pm_importance.importances_std,
                        'Mean Shapley values':mean_shapvalues}

pd.DataFrame(feature_importances).to_csv(outputdir + 
                                         "/GTB_brainage_meanfeatureimportances.csv")
pd.DataFrame(mean_interactions).to_csv(outputdir + 
                                       "/GTB_brainage_featureinteractions.csv")

# Save best-fit GTB model
joblib.dump(gtb_best,outputdir + "/GTB_brainage.sav")
        
    
