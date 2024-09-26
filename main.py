import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from lstm_model import BestModelEverLOL
from text_processor import LanguageModel

BATCH_SIZE = 32
NUM_EPOCHS = 5


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
    X_test = lm.preprocess(X_test)
    y_test = test_data['class'].values

    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    return X_test, y_test


def main():
    positive_file = "TrainingDataPositive.txt"
    negative_file = "TrainingDataNegative.txt"
    test_filepath = 'TestReviews.csv'

    all_reviews, all_labels = load_training_data(positive_file, negative_file, equal=True)

    # # best so far: window=1, vector_size=250 (not much difference), min_count=20
    # lm = LanguageModel(window=1, vector_size=250, min_count=20)
    # lm.train(all_reviews, epochs=10, check=True)
    # lm.save()
    lm = LanguageModel.load()

    X = lm.preprocess(all_reviews)
    X_train, X_val, y_train, y_val = train_test_split(X, np.array(all_labels), test_size=0.2)

    X_test, y_test = load_testing_data(test_filepath, lm)
    embedding_matrix = lm.get_embedding_matrix
    max_sequence_length = lm.get_max_sequence_length

    # Создание и обучение модели LSTM
    model = BestModelEverLOL(embedding_matrix, max_sequence_length, BATCH_SIZE)

    parts = 3
    part_len = len(X_train)//parts
    for i in range(parts):
        print(f"Part {i+1}/{parts}")

        X_part = X_train[part_len * i:part_len * (i + 1)]
        y_part = y_train[part_len * i:part_len * (i + 1)]

        model.train(X_part, y_part, X_val, y_val, NUM_EPOCHS)

        # loss, accuracy = model.evaluate(X_test, y_test)
        # print(f"Test Accuracy: {accuracy * 100:.2f}")
        model.show_confision_matrix(X_test, y_test)
        # model.show_roc_curve(X_test, y_test)

    model.save('lstm_model.keras')
    # model = BestModelEverLOL.load('lstm_model.keras')
    print(model.model.summary())



if __name__ == "__main__":
    main()
