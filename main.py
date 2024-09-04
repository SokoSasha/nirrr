import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from lstm_model import BestModelEverLOL
from text_processor import LanguageModel

BATCH_SIZE = 32
NUM_EPOCHS = 5


def load_training_data(positive_file, negative_file):
    with open(positive_file, 'r', encoding='utf-8') as f:
        positive_reviews = [line.strip() for line in f.readlines()]

    with open(negative_file, 'r', encoding='utf-8') as f:
        negative_reviews = [line.strip() for line in f.readlines()]

    positive_labels = [1] * len(positive_reviews)
    negative_labels = [0] * len(negative_reviews)

    all_reviews = positive_reviews + negative_reviews
    all_labels = positive_labels + negative_labels

    all_reviews, all_labels = shuffle(all_reviews, all_labels, random_state=42)

    return all_reviews, all_labels


def cut_to_size(X, all_labels):
    X_train, X_val, y_train, y_val = train_test_split(X, np.array(all_labels), test_size=0.2, random_state=42)

    train_size = (X_train.shape[0] // BATCH_SIZE) * BATCH_SIZE
    val_size = (X_val.shape[0] // BATCH_SIZE) * BATCH_SIZE

    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    X_val = X_val[:val_size]
    y_val = y_val[:val_size]

    return X_train, y_train, X_val, y_val


def load_testing_data(test_filepath, lm):
    test_data = pd.read_csv(test_filepath)
    X_test = test_data['review'].values
    X_test = lm.preprocess(X_test)
    y_test = test_data['class'].values

    test_size = (X_test.shape[0] // BATCH_SIZE) * BATCH_SIZE
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    return X_test, y_test


def main():
    positive_file = "TrainingDataPositive.txt"
    negative_file = "TrainingDataNegative.txt"
    test_filepath = 'TestReviews.csv'

    all_reviews, all_labels = load_training_data(positive_file, negative_file)

    # best so far: window=7, vector_size=200 (not much difference), min_count=10
    # lm = LanguageModel(window=7, vector_size=20, min_count=10)
    # lm.train(all_reviews, epochs=10, check=True)
    # lm.save()
    lm = LanguageModel.load()

    X = lm.preprocess(all_reviews)
    X_train, y_train, X_val, y_val = cut_to_size(X, all_labels)

    X_test, y_test = load_testing_data(test_filepath, lm)

    embedding_matrix = lm.get_embedding_matrix
    max_sequence_length = lm.get_max_sequence_length

    # Создание и обучение модели LSTM
    model = BestModelEverLOL(embedding_matrix, max_sequence_length, BATCH_SIZE)
    model.train(X_train, y_train, X_val, y_val, NUM_EPOCHS)
    model.save('lstm_model.keras')
    # model = BestModelEverLOL.load('lstm_model.keras')

    # Оценка модели
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}")

    model.show_confision_matrix(X_test, y_test, show_description=True)
    model.show_roc_curve(X_test, y_test)


if __name__ == "__main__":
    main()
