#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:01:18 2022
@author: martinsmarksgatavins
"""

#%% Package import
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import sys
import joblib
import time

#%% Data parsing
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
outputdir = sys.argv[3]
i = sys.argv[4]

X_train = train.drop(columns=["sub","age"],axis=1)
X_test = test.drop(columns=["sub","age"],axis=1).to_numpy()

features = X_train.columns.values

X_train = X_train.to_numpy()

y_train = train['age'].to_numpy().ravel()
y_test = test['age'].to_numpy().ravel()

test_ids = test['sub'].to_numpy().ravel()

#%% Main test_train_split & fold setup

outer_kf = KFold(n_splits=5,shuffle=True,random_state=20)
inner_kf = KFold(n_splits=10,shuffle=True,random_state=20)

#%% Setting up parameters to test for GridSearchCV
parameters = {"n_estimators":[100, 250, 500, 1000],
               "eta":[0.01, 0.025, 0.05, 0.1],
               "max_depth":[4, 6, 8, 10, 15],
               "subsample":[0.25, 0.5, 0.75, 1]}

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
with joblib.parallel_backend('multiprocessing'):
    modelTrain.fit(X_train,y_train)
    
y_pred = modelTrain.predict(X_test)

ba_ca_results = pd.DataFrame({'sub':test_ids,
                              'pred_age':y_pred,
                              'real_age':y_test,
                              'age_gap':y_pred-y_test})

ba_ca_results.to_csv(outputdir + "/brainage_" + i + ".csv",index=False)
print("Ending time in minutes:",(time.time()-start)/60.0)
        
    
