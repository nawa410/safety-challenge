import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from matplotlib import pyplot as plt
from data_preprocessor import DataPreprocessor
from sklearn.metrics import roc_curve,auc
import matplotlib.patches as patches
from scipy import interp
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgbm
import xgboost as xgb

def cross_validate(model, kfold, X, Y) :
    # plot arrows
    fig1 = plt.figure(figsize=[12,12])
    ax1 = fig1.add_subplot(111,aspect = 'equal')
    ax1.add_patch( patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5) )
    ax1.add_patch( patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5) )
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    for train, test in kfold.split(X, Y):
        prediction = model.fit(X.iloc[train],Y.iloc[train]).predict_proba(X.iloc[test])
        fpr, tpr, t = roc_curve(Y[test], prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
        i=i+1
    
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.4f )' % (mean_auc),lw=2, alpha=1)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    plt.show()
    
#### MAIN ######################################

import time
start_time = time.time()

features = pd.read_csv('data/processed/features_1560613343.csv')
label = pd.read_csv('data/processed/label_cleaned_1560613042.csv')

data = pd.merge(features, label, on='bookingID', how='inner')
data = data.drop("bookingID", axis=1)

X = data.loc[:, data.columns != 'label']
Y = data['label']

rf_model = RandomForestClassifier(
    n_jobs=4,
    n_estimators=500,
    max_depth=10
)

xgb_model = xgb.XGBClassifier(
    n_jobs=4,
    n_estimators=400,
    max_depth=2,
    colsample_bytree=0.8695,
    gamma=0.0216
)

lgbm_model = lgbm.LGBMClassifier(
    n_jobs=4,
    n_estimators=300,
    num_leaves=90,
    colsample_bytree=0.4484,
    max_depth=2
)

eclf = VotingClassifier(estimators=[
    ('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model)], voting='soft')

kfold = StratifiedKFold(n_splits=10, random_state=21)

cross_validate(eclf, kfold, X, Y)

print("--- %s seconds ---" % (time.time() - start_time))
