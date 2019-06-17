import pickle
import pandas as pd
from data_preprocessor import DataPreprocessor
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import calendar
import time

files = []
for r, d, f in os.walk('data/raw/features/'):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))
features = []
for f in files:
    df = pd.read_csv(f)
    features.append(df)
features = pd.concat(features)
features = features.sort_values(by=['bookingID', 'second'])

dp = DataPreprocessor()
features = dp.feature_engineering(features)
#features = pd.read_csv('data/processed/features_1560688534.csv')

files = []
for r, d, f in os.walk('data/raw/labels/'):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))
labels = []
for f in files:
    df = pd.read_csv(f)
    labels.append(df)
true_values_exist = True
if(len(labels) == 0):
    true_values_exist = False

if(true_values_exist) :
    labels = pd.concat(labels)
    labels = labels.sort_values(by=['bookingID'])
    features = pd.merge(features, labels, on='bookingID', how='inner')

bookingID = features["bookingID"]
features = features.drop("bookingID", axis=1)
X = features.loc[:, features.columns != 'label']
if(true_values_exist) :
    Y = features['label']

model = pickle.load(open('models/rf_xgb_lgbm_ensemble.sav', 'rb'))

predictions = model.predict_proba(X)
predictions = predictions[:, 1]
df_predictions = pd.DataFrame({ 'bookingID': bookingID, 'label': predictions })
ts = calendar.timegm(time.gmtime())
df_predictions.to_csv("output/predictions_"+str(ts)+".csv", index=False)
print("Predicted values is saved in: 'output/predictions_"+str(ts)+".csv'")

if(true_values_exist) :
    score = roc_auc_score(Y, predictions)
    print("AUC-ROC: "+str(score))