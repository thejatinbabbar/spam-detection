# importing modules
import os
import pandas as pd
from feature_selection import get_train_and_test_set
from sklearn.ensemble import GradientBoostingClassifier

def predict(data_path):
    """
    Choose 
    1. the best model (Gradient Boosting Classfier) 
    2. the best feature group (feature group 6)
    to fit the model and make predictions for test set
    """
    print('Making predictions for test set ...')
    test_file = os.path.join(data_path,'users','coded_ids_labels_test.csv')
    test_df = pd.read_csv(test_file)

    train_set, test_set = get_train_and_test_set(data_path)
    train = train_set[5]
    test = test_set[5]

    model = GradientBoostingClassifier(n_estimators=200)

    model.fit(train.drop('label',axis=1), train['label'])
    preds = model.predict(test.drop('label', axis=1))
    test_df['label'] = preds

    print('Saving test file ...')
    test_file = os.path.join(data_path,'users','coded_ids_labels_test_1.csv')
    test_df.to_csv(test_file, index=False)

    print('Done!')
    return