import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from lstm_model import BestModelEverLOL
from text_processor import LanguageModel

BATCH_SIZE = 32
NUM_EPOCHS = 5
MSL = 200



def load_training_data(positive_file, negative_file, equal=False):
    with open(positive_file, 'r', encoding='utf-8') as f:
        positive_reviews = [line.strip() for line in f.readlines()]

    with open(negative_file, 'r', encoding='utf-8') as f:
        negative_reviews = [line.strip() for line in f.readlines()]

    if equal:
        min_len = min(len(positive_reviews), len(negative_reviews))
        positive_reviews = positive_reviews[:min_len]
        negative_reviews = negative_reviews[:min_len]

    positive_labels = [1] * len(positive_reviews)
    negative_labels = [0] * len(negative_reviews)

    all_reviews = positive_reviews + negative_reviews
    all_labels = positive_labels + negative_labels

    all_reviews, all_labels = shuffle(all_reviews, all_labels)

    return all_reviews, all_labels


def load_testing_data(test_filepath, lm):
    test_data = pd.read_csv(test_filepath)
    X_test = test_data['review'].values
    X_test = lm.pad_sequences(lm.texts_to_sequences(X_test), maxlen=MSL)
    y_test = test_data['class'].values

    X_test, y_test = shuffle(X_test, y_test)

    return X_test, y_test


def main():
    positive_file = "TrainingDataPositive.txt"
    negative_file = "TrainingDataNegative.txt"
    test_filepath = 'TestReviews.csv'

    all_reviews, all_labels = load_training_data(positive_file, negative_file, equal=False)

    lm_window = 5
    lm_vector_size = 300
    lm_min_count = 20
    lm_epoches = 10

    class_weight = {0: 3.0, 1: 1.0}

    # Лучшие параметры на данный момент: window=5, vector_size=300, min_count=10
    # lm = LanguageModel(window=lm_window, vector_size=lm_vector_size, min_count=lm_min_count)
    # lm.train(all_reviews, epochs=lm_epoches, check=True)
    # lm.save()
    lm = LanguageModel.load()

    print("Preprocessing texts...", end="")
    start_time = time.perf_counter()
    X = lm.pad_sequences(lm.texts_to_sequences(all_reviews), maxlen=MSL)
    elapsed_time = time.perf_counter() - start_time
    print(f"done in {elapsed_time:0.4f} seconds")

    X_train, X_val, y_train, y_val = train_test_split(X, np.array(all_labels), test_size=0.2)
    X_test, y_test = load_testing_data(test_filepath, lm)

    # Так как модель stateful нужно, чтобы размеры всех батчей был одинаковым
    crop_size = len(X_train) // BATCH_SIZE * BATCH_SIZE
    X_train = X_train[:crop_size]
    y_train = y_train[:crop_size]

    crop_size = len(X_val) // BATCH_SIZE * BATCH_SIZE
    X_val = X_val[:crop_size]
    y_val = y_val[:crop_size]

    crop_size = len(X_test) // BATCH_SIZE * BATCH_SIZE
    X_test = X_test[:crop_size]
    y_test = y_test[:crop_size]

    # Создание и обучение модели LSTM
    embedding_matrix = lm.get_embedding_matrix
    model = BestModelEverLOL(embedding_matrix, MSL, BATCH_SIZE)

    parts = 4
    part_len = len(X_train)//parts
    for i in range(parts):
        print(f"Part {i+1}/{parts}")

        X_part = X_train[part_len * i:part_len * (i + 1)]
        y_part = y_train[part_len * i:part_len * (i + 1)]

        crop_size = len(X_part) // BATCH_SIZE * BATCH_SIZE
        X_part = X_part[:crop_size]
        y_part = y_part[:crop_size]

        model.train(X_part, y_part, X_val, y_val, NUM_EPOCHS, class_weight)
        y_pred = model.predict(X_test)
        model.reset()
        y_pred = (y_pred > 0.5).astype(int)
        model.show_confision_matrix(y_pred, y_test, title=f'Confusion matrix: training {i+1}/{parts}')
        model.print_metrics(y_pred, y_test)

    # model.train(X_train, y_train, X_val, y_val, NUM_EPOCHS, class_weight)

    # Метрики
    # y_pred = model.predict(X_test)
    # y_pred = (y_pred > 0.5).astype(int)
    # model.show_confision_matrix(y_pred, y_test,
    #                             title=f"w: {lm_window}, vs: {lm_vector_size}, mc: {lm_min_count}, e: {lm_epoches}, cw: {class_weight}, +precision +recall")
    # model.show_roc_curve(X_test, y_test)
    # model.print_metrics(y_pred, y_test)

    # model.save('lstm_model.keras')
    # model = BestModelEverLOL.load('lstm_model.keras')
    # model.summary()


if __name__ == "__main__":
    main()
