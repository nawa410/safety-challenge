import pickle
from prepare_training_data import get_training_data

X, Y = get_training_data()
model = pickle.load(open('models/rf_xgb_lgbm_ensemble.sav', 'rb'))

result = model.score(X, Y)
print(result)