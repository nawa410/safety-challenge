import pandas as pd
import pickle
from final_model import get_model
from prepare_training_data import get_training_data

X, Y = get_training_data()

model = get_model()
model.fit(X, Y)

pickle.dump(model, open('models/rf_xgb_lgbm_ensemble.sav', 'wb'))