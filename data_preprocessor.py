import pandas as pd
import numpy as np
import calendar
import time
import math

class DataPreprocessor:
    
    def prepare_training_data(self, features_filenames, label_filename):
        frames = []
        for filename in features_filenames :
            df = pd.read_csv(filename)
            frames.append(df)       
        
        features = pd.concat(frames)
        features = features.sort_values(by=['bookingID', 'second'])
        features = features.reset_index(drop=True)
        
        label = pd.read_csv(label_filename)
        label = label.sort_values(by=['bookingID'])
        label = label.reset_index(drop=True)
        
        features, label = self.__cleaning(features, label)
        print('cleaning is done ...')
        
        features = self.feature_engineering(features)   
     
        print('feature engineering is done ...')
    
    def __cleaning(self, features, label):
        # remove row with Speed=0
        features = features.loc[features['Speed'] != -1]
        
        # remove row with the same bookingID but different label
        last_bookingID = ''
        last_label = ''
        drop_bookingID = []
        for i in range(len(label)):
            if label.at[i, "bookingID"]==last_bookingID and label.at[i, "bookingID"]!=last_label:
                drop_bookingID.append(label.at[i, "bookingID"])
                
            last_bookingID = label.at[i, "bookingID"]
            last_label = label.at[i, "bookingID"]
            
        drop_index_label = []
        for i in range(len(drop_bookingID)):
            for j in range(len(label)):
                if label.at[j, "bookingID"]==drop_bookingID[i]:
                    drop_index_label.append(j)
            
        label = label.drop(drop_index_label)
        
        features = features.reset_index(drop=True)
        label = label.reset_index(drop=True)
        
        ts = calendar.timegm(time.gmtime())
        features.to_csv('data/interim/features_cleaned_'+str(ts)+'.csv', encoding='utf-8', index=False)
        label.to_csv('data/processed/label_cleaned_'+str(ts)+'.csv', encoding='utf-8', index=False)
        print('data/processed/label_cleaned_'+str(ts)+'.csv')

        return features, label
    
    def feature_engineering(self, features):
        features = features.sort_values(by=['bookingID', 'second'])
        features = features.drop("second", axis=1)
        features = features.reset_index(drop=True)
        
        # it's faster to process if dataframe is converted to numpy array first
        data = features.values
        data = self.__calculate_resultan(data)
        data = self.__calculate_changes_overtime(data)
        
        column_names = list(features.columns.values)
        column_names.append('acc_res')
        column_names.append('gyro_res')
        
        length = len(column_names)
        for i in range(length):
            if column_names[i] == 'bookingID':
                continue
            column_names.append(column_names[i]+'_changes')
            
        features = pd.DataFrame(data=data, columns=column_names)
        ts = calendar.timegm(time.gmtime())
        #features.to_csv('data/interim/features_res_changes_'+str(ts)+'.csv', encoding='utf-8', index=False)
        
        features_agg = self.__aggregate(data, column_names)
        #features_agg.to_csv('data/interim/features_aggregate_'+str(ts)+'.csv', encoding='utf-8', index=False)
        
        features_agg = self.__calculate_std(data, features_agg, column_names)
        #features_agg.to_csv('data/interim/features_aggregate_std_'+str(ts)+'.csv', encoding='utf-8', index=False)
        
        # drop column with prefix_name: sum_
        for c in column_names:
            if(c[:3] == 'sum'):
                features_agg = features_agg.drop(c, axis=1)
        
        features_agg.to_csv('data/processed/features_'+str(ts)+'.csv', encoding='utf-8', index=False)
        print('Features: data/processed/features_'+str(ts)+'.csv')

        return features_agg
        
    def __calculate_resultan(self, data):
        m, n = data.shape
        # extra columns for acc and gyro resultan
        extra_cols = np.zeros([m, 2], dtype = float)
        data = np.append(data, extra_cols, axis=1)
        
        # 10 : acceleration resultan
        # 11 : gyro resultan
        for i in range(m):
            if i%1000000 == 0 :
                print('calculate resultan : '+str(i)+' / '+str(m))
            data[i][10] = math.sqrt((data[i][3]*data[i][3]) + (data[i][4]*data[i][4]) + (data[i][5]*data[i][5]))
            data[i][11] = math.sqrt((data[i][6]*data[i][6]) + (data[i][7]*data[i][7]) + (data[i][8]*data[i][8]))
        
        return data
    
    def __calculate_changes_overtime(self, data):
        m, n = data.shape
         # extra columns, 11: all columns except bookingID
        extra_cols = np.zeros([m, 11], dtype = float)
        data = np.append(data, extra_cols, axis=1)
        
        last_trip = ''
        for i in range(m):
            if i%1000000 == 0 :
                print('calculate changes overtime : '+str(i)+' / '+str(m))
            if data[i][0] == last_trip :
                j = 12
                while j <= 22 :
                    data[i][j] = abs(data[i][j-11] - data[i-1][j-11])

                    # special case for Bearing_changes.. 
                    if j == 13:
                        if data[i][j] >= 180:
                            data[i][j] = 360 - data[i][j]
            
                    j += 1
            last_trip = data[i][0]
        
        return data
         
    def __aggregate(self, data, column_names):
        agg_columns = ['bookingID', 'n_rows']
        for c in column_names:
            if c == 'bookingID':
                continue
            agg_columns.append('min_'+c)
            agg_columns.append('max_'+c)
            agg_columns.append('sum_'+c)
            agg_columns.append('mean_'+c)
            agg_columns.append('std_'+c)
            agg_columns.append('range_'+c)
            
        #features_agg = pd.DataFrame(columns=agg_columns)
        features_agg_arr = []
        
        save = {}
        for c in column_names:
            if c == 'bookingID':
                continue
            save['min_'+c]= 99999999.0
            save['max_'+c]=-99999999.0
            save['sum_'+c]=0.0
        
        n_rows = 0.0
        ind = 0
        last_trip = ''
        first = True
        
        m, n = data.shape
        for i in range(m):
            if i%1000000 == 0 :
                print('calculate aggregate : '+str(i)+' / '+str(m))
            
            if data[i][0] != last_trip and first==False :
                row = [last_trip, n_rows]
                for c in column_names:
                    if c == 'bookingID':
                        continue
                    row.append(save['min_'+c])
                    row.append(save['max_'+c])
                    row.append(save['sum_'+c])
                    row.append(save['sum_'+c]/n_rows)
                    row.append(0.00)
                    row.append(save['max_'+c]-save['min_'+c])
                #features_agg.loc[ind] = row
                features_agg_arr.append(row)
                for c in column_names:
                    if c == 'bookingID':
                        continue
                    save['min_'+c]= 99999999.0
                    save['max_'+c]=-99999999.0
                    save['sum_'+c]=0.0
                n_rows = 0.0
                ind+=1
            
            last_trip=data[i][0]
            for j in range(len(column_names)):
                if column_names[j] == 'bookingID':
                    continue
                save['min_'+column_names[j]]=min(save['min_'+column_names[j]], data[i][j])
                save['max_'+column_names[j]]=max(save['max_'+column_names[j]], data[i][j])
                save['sum_'+column_names[j]]=save['sum_'+column_names[j]] + data[i][j]
            n_rows+=1
            first = False
        row = [last_trip, n_rows]
        for c in column_names:
            if c == 'bookingID':
                continue
            row.append(save['min_'+c])
            row.append(save['max_'+c])
            row.append(save['sum_'+c])
            row.append(save['sum_'+c]/n_rows)
            row.append(0.00)
            row.append(save['max_'+c]-save['min_'+c])
        #features_agg.loc[ind] = row
        features_agg_arr.append(row)
        features_agg = pd.DataFrame(features_agg_arr, columns=agg_columns)

        return features_agg
    
    def __calculate_std(self, data, features_agg, column_names):
        
        save = {}
        for c in column_names:
            if c == 'bookingID':
                continue
            save[c]=0.0
        
        first = True
        last_trip = ''
        j = 0
        k = 0
        m, n = data.shape
        features_agg_arr = features_agg.values
        agg_columns = list(features_agg.columns.values)
        for i in range(m):
            if i%1000000 == 0 :
                print('calculate std : '+str(i)+' / '+str(m))
                
            k += 1
                
            if data[i][0] != last_trip and first==False : 
                for k in range(len(column_names)):
                    if column_names[k] == 'bookingID':
                        continue
                    #features_agg.at[j, 'std_'+column_names[k]] = math.sqrt(save[column_names[k]]/features_agg.at[j, 'n_rows'])        
                    features_agg_arr[j][k*6] = math.sqrt(save[column_names[k]]/features_agg_arr[j][1])  
                for c in column_names:
                    if c == 'bookingID':
                        continue
                    save[c]=0.0    
                j += 1
                k = 0        
                
            last_trip = data[i][0]
            for k in range(len(column_names)):
                if column_names[k] == 'bookingID':
                    continue
                #save[column_names[k]] += (data[i][k]-features_agg.at[j, 'mean_'+column_names[k]]) * (data[i][k]-features_agg.at[j, 'mean_'+column_names[k]])
                save[column_names[k]] += (data[i][k]-features_agg_arr[j][5*k+(k-1)]) * (data[i][k]-features_agg_arr[j][5*k+(k-1)])
            first = False
        
        for k in range(len(column_names)):
            if column_names[k] == 'bookingID':
                continue
            #features_agg.at[j, 'std_'+column_names[k]] = math.sqrt(save[column_names[k]]/features_agg.at[j, 'n_rows'])
            features_agg_arr[j][k*6] = math.sqrt(save[column_names[k]]/features_agg_arr[j][1])
        
        features_agg = pd.DataFrame(features_agg_arr, columns=agg_columns)

        return features_agg
        
        