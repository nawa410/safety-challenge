#%%
import pandas as pd
import numpy as np
import time
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm

#%%
features = pd.read_csv('data/processed/features_1560613343.csv')
label = pd.read_csv('data/processed/label_cleaned_1560613042.csv')

data = pd.merge(features, label, on='bookingID', how='inner')
data = data.drop("bookingID", axis=1)
X = data.loc[:, data.columns != 'label']
Y = data['label']

#%%
start_time = time.time()

def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth'])
    }
    clf = RandomForestClassifier(n_jobs=6, **params)
    score = cross_val_score(clf, X, Y, scoring='roc_auc', cv=StratifiedKFold(n_splits=5)).mean()
    print("ROC-AUC {:.3f} params {}".format(score, params))

    return {'loss':1-score, 'status': STATUS_OK }

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'max_depth': hp.quniform('max_depth', 1, 10, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)

print("Random Forest: Hyperopt estimated optimum {}".format(best))
print("--- %s seconds ---" % (time.time() - start_time))

#%%
start_time = time.time()

def objective(params):
    params = {
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth'])
    }
    clf = xgb.XGBClassifier(n_jobs=6, **params)    
    score = cross_val_score(clf, X, Y, scoring='roc_auc', cv=StratifiedKFold(n_splits=5)).mean()
    print("ROC-AUC {:.3f} params {}".format(score, params))

    return {'loss':1-score, 'status': STATUS_OK }

space = {
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'max_depth': hp.quniform('max_depth', 1, 10, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)

print("XGBoost: Hyperopt estimated optimum {}".format(best))

print("--- %s seconds ---" % (time.time() - start_time))

#%%
start_time = time.time()

def objective(params):
    params = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth'])
    }    
    clf = lgbm.LGBMClassifier(n_jobs=6, **params)    
    score = cross_val_score(clf, X, Y, scoring='roc_auc', cv=StratifiedKFold(n_splits=5)).mean()
    print("ROC-AUC {:.3f} params {}".format(score, params))

    return {'loss':1-score, 'status': STATUS_OK }

space = {
    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'max_depth': hp.quniform('max_depth', 1, 10, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)

print("LightGBM: Hyperopt estimated optimum {}".format(best))
print("--- %s seconds ---" % (time.time() - start_time))

#%%
start_time = time.time()

rf_model = RandomForestClassifier(
    n_jobs=4,
    n_estimators=500,
    max_depth=10
)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    n_jobs=4,
    max_depth=2,
    colsample_bytree=0.8695,
    gamma=0.0216
)

lgbm_model = lgbm.LGBMClassifier(
    n_estimators=300,
    num_leaves=90,
    colsample_bytree=0.4484,
    max_depth=2
)

models = [
    ('Random Forest', rf_model),
    ('XGBoost', xgb_model),
    ('LightGBM', lgbm_model),
]

for label, model in models:
    scores = cross_val_score(model, X, Y, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
    print("ROC AUC: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))

print("--- %s seconds ---" % (time.time() - start_time))
#%%
