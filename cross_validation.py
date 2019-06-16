import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc
import matplotlib.patches as patches
from scipy import interp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from model_configuration import get_model
from prepare_training_data import get_training_data

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

def main():
    X, Y = get_training_data()

    model = get_model()

    kfold = StratifiedKFold(n_splits=10, random_state=21)
    cross_validate(model, kfold, X, Y)


if __name__ == '__main__':
    main()