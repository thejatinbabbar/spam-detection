# importing modules
import pandas as pd
import numpy as np
from sklearn import set_config
from feature_selection import get_train_and_test_set
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

set_config(print_changed_only=True)

def get_crossval_scores(data_path):
    """
    Perform cross validation on the dataset and fit on different models
    """

    models = [KNeighborsClassifier(n_neighbors=5),
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              RandomForestClassifier(n_estimators=200),
              GradientBoostingClassifier(),
              GradientBoostingClassifier(n_estimators=200)]
    train_set, _ = get_train_and_test_set(data_path)

    scores = {}
    for model in models:
        scores[model] = {}

    print('Cross Validation Score')

    for model in models:
        print(model)
        for i, train in enumerate(train_set, 1):
            X, y = train.drop('label', axis=1), train['label']
            cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
            score = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            scores[model][f'group {i}'] = score
            print('---- feature group', i, ':', score, 'Mean:', np.mean(score))
        print()

    return scores