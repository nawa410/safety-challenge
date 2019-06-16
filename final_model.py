from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Using 3 models with optimal hyper parameters given this problem

def get_model():
    
    rf = RandomForestClassifier(
        n_jobs=4,
        n_estimators=500,
        max_depth=10
    )

    xgb = XGBClassifier(
        n_jobs=4,
        n_estimators=400,
        max_depth=2,
        colsample_bytree=0.8695,
        gamma=0.0216
    )

    lgbm = LGBMClassifier(
        n_jobs=4,
        n_estimators=300,
        num_leaves=90,
        colsample_bytree=0.4484,
        max_depth=2
    )

    VC = VotingClassifier(estimators=[
        ('rf', rf), 
        ('xgb', xgb), 
        ('lgbm', lgbm)], voting='soft')

    return VC