import pandas as pd

features = pd.read_csv('data/interim/features_cleaned_1560667948.csv')
label = pd.read_csv('data/processed/label_cleaned_1560667948.csv')

data = pd.merge(features, label, on='bookingID', how='inner')
features = data.loc[:, data.columns != 'label']
label = data.loc[data.groupby(["bookingID"])["label"].idxmax()] 

features = features.reset_index(drop=True)
label = label.reset_index(drop=True)


train_length = 0.5 * len(label)

y_train = label[:int(train_length)]
y_test = label[-int(len(label) - train_length + 1):]

X_train = features[features['bookingID'].isin(y_train.bookingID)]
X_test = features[features['bookingID'].isin(y_test.bookingID)]

y_train = y_train[['bookingID', 'label']]
y_test = y_test[['bookingID', 'label']]
X_test.to_csv('data/raw/features/features.csv', encoding='utf-8', index=False)
y_test.to_csv('data/raw/labels/labels.csv', encoding='utf-8', index=False)