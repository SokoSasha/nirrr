import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Embedding, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

batch_size = 32

# Загрузка данных
def load_data(positive_file, negative_file):
    with open(positive_file, 'r', encoding='utf-8') as f:
        positive_reviews = [line.strip() for line in f.readlines()]

    with open(negative_file, 'r', encoding='utf-8') as f:
        negative_reviews = [line.strip() for line in f.readlines()]

    positive_labels = [1] * len(positive_reviews)
    negative_labels = [0] * len(negative_reviews)

    all_reviews = positive_reviews + negative_reviews
    all_labels = positive_labels + negative_labels

    return all_reviews, all_labels


# Предобработка текста
def preprocess_text(reviews):
    processed_reviews = [review.lower().strip() for review in reviews]
    return processed_reviews


# Векторизация текста с использованием Word2Vec
def train_word2vec_model(reviews, vector_size=300, window=5, min_count=1):
    tokenized_reviews = [review.split() for review in reviews]
    word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return word2vec_model


# Преобразование текста в последовательности
def text_to_sequences(reviews, word2vec_model, max_sequence_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)

    sequences = tokenizer.texts_to_sequences(reviews)
    word_index = tokenizer.word_index

    embedding_matrix = np.zeros((len(word_index) + 1, word2vec_model.vector_size))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    sequences_padded = pad_sequences(sequences, maxlen=max_sequence_length)

    return sequences_padded, embedding_matrix, tokenizer


# Создание модели LSTM
def create_lstm_model(embedding_matrix, max_sequence_length, batch_size):
    vocab_size, embedding_dim = embedding_matrix.shape
    model = Sequential()

    model.add(InputLayer(batch_input_shape=(batch_size, max_sequence_length,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=False))
    model.add(LSTM(100,
                   stateful=True,
                   return_sequences=False,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   unroll=True))

    # Добавление слоя Dense
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_and_preprocess_test_data(filepath, tokenizer, max_sequence_length, batch_size):
    # Загрузка тестовых данных
    test_data = pd.read_csv(filepath)

    # Разделение на признаки и метки
    X_test = test_data['review'].values
    y_test = test_data['class'].values

    # Преобразование текстов в последовательности
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_sequence_length)

    # Убедитесь, что размер тестовых данных кратен размеру батча
    test_size = (X_test.shape[0] // batch_size) * batch_size
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    return X_test, y_test

positive_file = "TrainingDataPositive.txt"
negative_file = "TrainingDataNegative.txt"

# Загрузка данных
reviews, labels = load_data(positive_file, negative_file)

# Предобработка текста
processed_reviews = preprocess_text(reviews)

# Загрузка модели Word2Vec
word2vec_model = Word2Vec.load('word2vec_model.bin')

# Преобразование текста в последовательности
max_sequence_length = 100
X, embedding_matrix, tokenizer = text_to_sequences(processed_reviews, word2vec_model, max_sequence_length)

# Преобразование меток
y = np.array(labels)

# Разделение данных на обучающую и тестовую выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

test_filepath = 'TestReviews.csv'
# Загрузка и подготовка тестовых данных
X_test, y_test = load_and_preprocess_test_data(test_filepath, tokenizer, max_sequence_length, batch_size)

# Убедитесь, что размеры данных кратны размеру батча
train_size = (X_train.shape[0] // batch_size) * batch_size
test_size = (X_test.shape[0] // batch_size) * batch_size
val_size = (X_val.shape[0] // batch_size) * batch_size

X_train = X_train[:train_size]
y_train = y_train[:train_size]

X_val = X_val[:val_size]
y_val = y_val[:val_size]

X_test = X_test[:test_size]
y_test = y_test[:test_size]

# Создание и обучение модели LSTM
model = create_lstm_model(embedding_matrix, max_sequence_length, batch_size)

# Определение контрольной точки для сохранения лучшей модели
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Обучение модели
history = model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_val, y_val), shuffle=False, callbacks=[checkpoint])

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Test Accuracy: {accuracy}")
