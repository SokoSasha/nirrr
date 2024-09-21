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

    all_reviews, all_labels = load_training_data(positive_file, negative_file)

    # # best so far: window=7, vector_size=200 (not much difference), min_count=10
    # lm = LanguageModel(window=7, vector_size=20, min_count=10)
    # lm.train(all_reviews, epochs=10, check=True)
    # lm.save()
    lm = LanguageModel.load()

    X = lm.preprocess(all_reviews)
    X_train, X_val, y_train, y_val = train_test_split(X, np.array(all_labels), test_size=0.2, random_state=42)

    X_train_0 = X_train[:len(X_train)//2]
    X_train_1 = X_train[len(X_train)//2:]

    y_train_0 = y_train[:len(y_train)//2]
    y_train_1 = y_train[len(y_train)//2:]

    X_test, y_test = load_testing_data(test_filepath, lm)

    embedding_matrix = lm.get_embedding_matrix
    max_sequence_length = lm.get_max_sequence_length

    # Создание и обучение модели LSTM
    model = BestModelEverLOL(embedding_matrix, max_sequence_length, BATCH_SIZE)
    model.train(X_train_0, y_train_0, X_val, y_val, NUM_EPOCHS)
    # model.save('lstm_model.keras')
    # model = BestModelEverLOL.load('lstm_model.keras')
    # print(model.model.summary())

    # Оценка модели
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}")

    model.show_confision_matrix(X_test, y_test)
    model.show_roc_curve(X_test, y_test)

    reviews = [
        "The location is in a strip mall, but this place is a diamond in the rough.  The food was some of the best italian food I have had anywhere.  I live in Kansas City, which has a pretty large and long standing italian populaiton.  The food is a good as any of the old italian places in KC.  Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "I live in Kansas City, which has a pretty large and long standing italian population. Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "Never going back. This was my second time here. Wasn't that impressed the first time, but thought I'll give it another look, more because one of the guys in our team really wanted to go here. Food: Just ok, over-priced and very small portions. Came out hungry. Deserts are quite nice. Service: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed. Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed.",
        "Food: Just ok, over-priced and very small portions. Deserts are quite nice. Service: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed. Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed.",
    ]

    warmup = lm.preprocess(["The location is in a strip mall, but this place is a diamond in the rough."])
    model.predict(warmup, verbose=0)

    total_time = 0
    total_sentences = 0

    for review in reviews:
        conf = []
        times = []
        print(f'Review: {review}')
        review = review.split('.')
        context_buffer = []
        for sentence in review:
            if sentence.strip():
                context_buffer = ['.'.join(context_buffer + [sentence])]
                preprocessed_sentence = lm.preprocess(context_buffer)
                start_time = time.perf_counter()
                confidence = model.predict(preprocessed_sentence, verbose=0)[0][0]
                conf.append(confidence)
                elapsed_time = time.perf_counter() - start_time
                times.append(elapsed_time)
                total_time += elapsed_time
                total_sentences += 1
                # print(f"{elapsed_time:.4f} seconds")
                # print(f"{confidence * 100:.2f}% positive")

        plt.figure(figsize=(12, 5))

        # График уверенности
        plt.subplot(1, 2, 1)
        plt.plot(conf, marker='o', color='b')
        plt.title('Confidence Levels')
        plt.xlabel('Sentence Index')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.grid()

        # График времени
        plt.subplot(1, 2, 2)
        plt.plot(times, marker='o', color='r')
        plt.title('Processing Times')
        plt.xlabel('Sentence Index')
        plt.ylabel('Time (seconds)')
        plt.ylim(0, 0.1)
        plt.grid()

        plt.tight_layout()
        plt.show()

        print('-----------------------------------------------------------------------------')

    print(f'Average processing time per sentence: {total_time/total_sentences:.4f} seconds')


    model.train(X_train_1, y_train_1, X_val, y_val, NUM_EPOCHS)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}")

    model.show_confision_matrix(X_test, y_test)
    model.show_roc_curve(X_test, y_test)

    total_time = 0
    total_sentences = 0

    for review in reviews:
        conf = []
        times = []
        print(f'Review: {review}')
        review = review.split('.')
        context_buffer = []
        for sentence in review:
            if sentence.strip():
                context_buffer = ['.'.join(context_buffer + [sentence])]
                preprocessed_sentence = lm.preprocess(context_buffer)
                start_time = time.perf_counter()
                confidence = model.predict(preprocessed_sentence, verbose=0)[0][0]
                conf.append(confidence)
                elapsed_time = time.perf_counter() - start_time
                times.append(elapsed_time)
                total_time += elapsed_time
                total_sentences += 1
                # print(f"{elapsed_time:.4f} seconds")
                # print(f"{confidence * 100:.2f}% positive")

        plt.figure(figsize=(12, 5))

        # График уверенности
        plt.subplot(1, 2, 1)
        plt.plot(conf, marker='o', color='b')
        plt.title('Confidence Levels')
        plt.xlabel('Sentence Index')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.grid()

        # График времени
        plt.subplot(1, 2, 2)
        plt.plot(times, marker='o', color='r')
        plt.title('Processing Times')
        plt.xlabel('Sentence Index')
        plt.ylabel('Time (seconds)')
        plt.ylim(0, 0.1)
        plt.grid()

        plt.tight_layout()
        plt.show()

        print('-----------------------------------------------------------------------------')

    print(f'Average processing time per sentence: {total_time / total_sentences:.4f} seconds')



if __name__ == "__main__":
    main()
