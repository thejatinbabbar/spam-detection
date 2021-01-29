# importing modules
from feature_selection import get_train_and_val_set
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train(model, X_train, y_train):
    """
    Fit the model and return the train score
    """
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)

    return score

def evaluate(model, X_val, y_val):
    """
    Evaluate the model on the validation set and return the val score
    """
    score = model.score(X_val, y_val)

    return score

def predict(model, X_test):
    """
    Predict the labels for the test set
    """
    preds = model.predict(X_test)
    
    return preds

data_path = '/content/drive/MyDrive/data_mining/Social_spammers_dataset'
models = [DecisionTreeClassifier(),
          RandomForestClassifier(),
          GradientBoostingClassifier()]
train_val_set = get_train_and_val_set(data_path)

scores = {}
for model in models:
    scores[type(model).__name__] = {'train_scores': [], 'val_scores': []}

for model in models:
    print(type(model).__name__)
    for i, (X_train, y_train, X_val, y_val) in enumerate(train_val_set, 1):
        score = train(model, X_train, y_train)
        scores[type(model).__name__]['train_scores'].append(score)
        print('---- train score', i, ':', score)

        score = evaluate(model, X_val, y_val)
        scores[type(model).__name__]['val_scores'].append(score)
        print('---- val score', i, ':', score)
        print()