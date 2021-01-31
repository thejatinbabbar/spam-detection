# importing modules
import pandas as pd
import numpy as np
from feature_selection import get_train_and_val_set
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

def get_crossval_scores(data_path, to_print=False):
    """
    Perform cross validation on the dataset and fit on different models
    """
    
    models = [DecisionTreeClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier()]
    _, train_set = get_train_and_val_set(data_path, return_full_set=True)

    scores = {}
    for model in models:
        scores[type(model).__name__] = {}

    for model in models:
        if to_print:
            print(type(model).__name__)
        for i, train in enumerate(train_set, 1):
            X, y = train.drop('label', axis=1), train['label']
            cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
            score = cross_val_score(model, X, y, cv=cv)
            scores[type(model).__name__][f'group {i}'] = score
            if to_print:
                print('---- feature group', i, ':', score)
                print()

    return scores

if __name__ == '__main__':
    data_path = '/content/drive/MyDrive/data_mining/Social_spammers_dataset'
    scores = get_crossval_scores(data_path, to_print=True)
    pd.DataFrame(scores).to_pickle('/content/drive/MyDrive/data_mining/notebooks/crossval_scores.pkl')
