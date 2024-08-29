import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer

# Загрузка предобученной модели и Word2Vec
model = load_model('best_model.keras')
word2vec_model = Word2Vec.load('word2vec_model.bin')  # Убедитесь, что сохранили модель Word2Vec
tokenizer = Tokenizer()
tokenizer.fit_on_texts([])  # Используйте текст, на котором обучали токенизатор


# Подготовка данных
def preprocess_text(reviews):
    return [review.lower().strip() for review in reviews]


def text_to_sequences(reviews, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences(reviews)
    sequences_padded = pad_sequences(sequences, maxlen=max_sequence_length)
    return sequences_padded


def process_review(review, model, tokenizer, word2vec_model, max_sequence_length):
    processed_review = preprocess_text([review])
    sequences_padded = text_to_sequences(processed_review, tokenizer, max_sequence_length)

    # Reset the model states before processing each review
    model.reset_states()

    # Make predictions in chunks (if necessary)
    prediction = model.predict(sequences_padded, batch_size=1)

    confidence = prediction[0][0]
    return confidence


# Пример использования
max_sequence_length = 100  # Убедитесь, что этот параметр соответствует вашему обучению

# Список отзывов для обработки
reviews = [
    "I absolutely love this product! It's fantastic.",
    "The quality is terrible, I'm very disappointed.",
    "Service was okay, but I expected better.",
    "Absolutely horrible experience, would not recommend."
]

for review in reviews:
    confidence = process_review(review, model, tokenizer, word2vec_model, max_sequence_length)
    print(f"Review: '{review}'")
    print(f"Confidence that the review is negative: {confidence:.4f}\n")
