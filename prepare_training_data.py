import pandas as pd
from data_preprocessor import DataPreprocessor

dp = DataPreprocessor()

features_filenames = []
for i in range(10) :
   features_filenames.append('data/raw/features/part-0000'+str(i)+'-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
label_filename = 'data/raw/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv'

dp.prepare_training_data(features_filenames, label_filename)