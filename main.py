# importing modules
from models_with_crossval import get_crossval_scores
from make_predictions import predict


if __name__ == '__main__':
    data_path = '/content/drive/MyDrive/data_mining/Social_spammers_dataset'
    scores = get_crossval_scores(data_path)
    predict(data_path)