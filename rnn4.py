import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint


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

    return sequences_padded, embedding_matrix


# Создание модели LSTM
def create_lstm_model(embedding_matrix, max_sequence_length, batch_size):
    vocab_size, embedding_dim = embedding_matrix.shape
    model = Sequential()

    model.add(InputLayer(batch_input_shape=(batch_size, max_sequence_length)))

    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        trainable=False))

    model.add(LSTM(100,
                   return_sequences=False,
                   stateful=True,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   unroll=True))

    # Добавление слоя Dense
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Основной код
positive_file = 'TrainingDataPositive.txt'
negative_file = 'TrainingDataNegative.txt'

# Загрузка данных
reviews, labels = load_data(positive_file, negative_file)

# Предобработка текста
processed_reviews = preprocess_text(reviews)

# Обучение модели Word2Vec
word2vec_model = train_word2vec_model(processed_reviews)

# Сохранение модели Word2Vec
word2vec_model.save('word2vec_model.bin')
# word2vec_model = Word2Vec.load('word2vec_model.bin')

# Преобразование текста в последовательности
max_sequence_length = 100
X, embedding_matrix = text_to_sequences(processed_reviews, word2vec_model, max_sequence_length)

# Преобразование меток
y = np.array(labels)

# Проверка размеров данных
print("X shape:", X.shape)
print("y shape:", y.shape)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Размер батча
batch_size = 32

# Создание и обучение модели LSTM
model = create_lstm_model(embedding_matrix, max_sequence_length, batch_size)

# Определение контрольной точки для сохранения лучшей модели
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Обучение модели
history = model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_split=0.2, shuffle=False, callbacks=[checkpoint])

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Сохранение embedding_matrix
np.save('embedding_matrix.npy', embedding_matrix)
