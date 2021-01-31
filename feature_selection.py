# importing modules
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

def import_files(data_path):
    """
    Import the files and join the features with the labels to obtain a single dataframe
    """
    users = os.path.join(data_path, 'users')
    users_features = os.path.join(data_path, 'users_features')

    coded_ids = pd.read_csv(os.path.join(users,'coded_ids.csv')).set_index('coded_id')
    coded_ids_labels_train = pd.read_csv(os.path.join(users,'coded_ids_labels_train.csv')).set_index('coded_id')
    coded_ids = coded_ids.join(coded_ids_labels_train)
    coded_ids.reset_index(inplace=True)
    coded_ids.set_index('user_id', inplace=True)

    features = pd.read_csv(os.path.join(users_features, 'features.csv')).set_index('user_id')
    # features_names = pd.read_csv(os.path.join(users_features, 'features_names.txt'), header=None)

    data = features.join(coded_ids)
    data.reset_index(inplace=True, drop=True)
    data.set_index('coded_id', inplace=True)
    data.sort_index(inplace=True)

    return data

def get_clean_data(data_path):
    """
    Clean the dataset by:

    1. encode categorical variable
    2. remove unnecessary features
    3. fill null values
    4. convert datetime to timestamp
    """
    data = import_files(data_path)

    encoder = LabelEncoder()
    data['lang'] = encoder.fit_transform(data['lang'])
    data['time_zone'] = encoder.fit_transform(data['time_zone'].astype(str))
    data['date_newest_tweet'] = pd.to_datetime(data['date_newest_tweet']).astype('int64') // 10 ** 9
    data['date_oldest_tweet'] = pd.to_datetime(data['date_oldest_tweet']).astype('int64') // 10 ** 9
    data['utc_offset'] = data['utc_offset'].fillna(0)

    cols_to_remove = ['avg_intertweet_times',
                      'max_intertweet_times',
                      'min_intertweet_times',
                      'std_intertweet_times',
                      'followers_count_minus_2002',
                      'friends_count_minus_2002',
                      'spam_in_screen_name']
    data.drop(cols_to_remove, axis=1, inplace=True)

    return data

def get_feature_groups(data):
    """
    Create different feature groups based on the correlation score with the target variable

    1. all columns
    2. columns with correlation score > 0.2
    3. columns with correlation score > 0.3
    4. top 30 columns with chi square score
    5. top 50 columns with chi square score
    6. top 80 columns with chi square score
    """
    groups = []
    groups.append(data.columns)
    groups.append(data.corr()[data.corr().label.abs() > 0.2].label.index.values)
    groups.append(data.corr()[data.corr().label.abs() > 0.3].label.index.values)
    
    scaler = MinMaxScaler()
    data_new = scaler.fit_transform(data[data.label.notnull()].drop('label', axis=1))
    selector = SelectKBest(chi2, k=30)
    selector.fit(data_new, data[data.label.notnull()]['label'])
    indices =[i for i, x in enumerate(selector.get_support()) if x]
    group = [data.columns[i] for i in indices]
    group.append('label')
    groups.append(group)

    selector = SelectKBest(chi2, k=50)
    selector.fit(data_new, data[data.label.notnull()]['label'])
    indices =[i for i, x in enumerate(selector.get_support()) if x]
    group = [data.columns[i] for i in indices]
    group.append('label')
    groups.append(group)

    selector = SelectKBest(chi2, k=80)
    selector.fit(data_new, data[data.label.notnull()]['label'])
    indices =[i for i, x in enumerate(selector.get_support()) if x]
    group = [data.columns[i] for i in indices]
    group.append('label')
    groups.append(group)

    return groups

def get_train_and_val_set(data_path, return_full_set=False):
    """
    Split the dataset into train set, validation set, and test set for each feature group
    """
    data = get_clean_data(data_path)
    groups = get_feature_groups(data)

    train_set = []
    train_val_set = []
    for columns in groups:

        train = data[columns][data.label.notnull()].copy()
        train_set.append(train)
        test = data[columns][data.label.isnull()].copy()
        train, val = train_test_split(train, test_size=86, random_state=101)

        X_train, y_train = train.drop('label', axis=1), train['label']
        X_val, y_val = val.drop('label', axis=1), val['label']

        train_val_set.append((X_train, y_train, X_val, y_val))

    if return_full_set:
        return train_val_set, train_set
    else:
        return train_val_set

if __name__ == '__main__':
    data_path = '/content/drive/MyDrive/data_mining/Social_spammers_dataset'
    t_v_set = get_train_and_val_set(data_path)