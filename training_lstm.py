
#%%
import pandas as pd

features = pd.read_csv('data/interim/features_cleaned_1560214342.csv')
label = pd.read_csv('data/interim/label_cleaned_1560214342.csv')


#%%
data = pd.merge(features, label, on='bookingID', how='inner')
features = data.loc[:, data.columns != 'label']
features


#%%
label = data.loc[data.groupby(["bookingID"])["label"].idxmax()] 
#label = pd.DataFrame(label, columns=["bookingID", 'label'])
label


#%%
features = features.reset_index(drop=True)
label = label.reset_index(drop=True)


#%%
train_length = 0.7 * len(label)
train_length

y_train = label[:int(train_length)]
y_test = label[-int(len(label) - train_length + 1):]

y_train2 = y_train['label']
y_test2 = y_test['label']

X_train = features[features['bookingID'].isin(y_train.bookingID)]
X_test = features[features['bookingID'].isin(y_test.bookingID)]


#%%
def toList(features, columns):
    data = features.values
    m, n = data.shape
    seq = []
    ls = []
    last = ''
    first = True
    for i in range(m):
        if i % 1000000 == 0:
            print(str(i)+" / "+str(m))
        if data[i][0] != last and first==False:
            seq.append(ls)
            ls = list()
        last = data[i][0]
        ls2=[]
        for c in columns:
            ls2.append(data[i][c])
        ls.append(ls2)
        first=False
    if len(ls) != 0:
        seq.append(ls)

    return seq

X_train2 = toList(X_train, [1, 2, 3, 4, 5, 6, 7, 8, 10])
X_test2 = toList(X_test, [1, 2, 3, 4, 5, 6, 7, 8, 10])


#%%
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
# fix random seed for reproducibility
np.random.seed(7)

max_len = 500
X_train2 = sequence.pad_sequences(X_train2, maxlen=max_len)
X_test2 = sequence.pad_sequences(X_test2, maxlen=max_len)

# create the model
model = Sequential()
model.add(LSTM(100, input_shape=(max_len, 9)))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train2, y_train2, epochs=100, batch_size=256)
# Final evaluation of the model
scores = model.evaluate(X_test2, y_test2, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))




#%%
