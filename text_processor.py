import pickle
import string
import time

import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class LanguageModel:
    def __init__(self, max_sequence_length=200, vector_size=300, window=10, min_count=5, workers=4):
        self.max_sequence_length = max_sequence_length
        self.model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        self.tokenizer = Tokenizer()
        self.embedding_matrix = None

    def train(self, reviews: list[str], epochs=5, update=False, check=False):
        print("Training model... ", end="")
        start_time = time.perf_counter()
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        processed_reviews = [review.lower().translate(translator).split() for review in reviews]

        self.tokenizer.fit_on_texts(processed_reviews)

        self.model.build_vocab(processed_reviews, update=update)
        history = self.model.train(processed_reviews, total_examples=len(reviews), epochs=epochs)

        elapsed_time = time.perf_counter() - start_time
        print(f"done in {elapsed_time:.4f} seconds")

        word_index = self.tokenizer.word_index
        self.embedding_matrix = np.zeros((len(word_index) + 1, self.model.vector_size))
        for word, i in word_index.items():
            if word in self.model.wv:
                self.embedding_matrix[i] = self.model.wv[word]

        if check:
            similarity = self.model.wv.similarity('good', 'great')
            print(f"Similarity between 'good' and 'great': {similarity * 100:.2f}%")

            most_similar_words = self.model.wv.most_similar('good', topn=5)
            print(f"Most similar words to 'good': {most_similar_words}")

            most_similar_words = self.model.wv.most_similar('bad', topn=5)
            print(f"Most similar words to 'bad': {most_similar_words}")

        return history

    def save(self, word2vec_name='word2vec_model.bin', embedding_name='embedding_matrix.npy', tokenizer_name='tokenizer.pkl'):
        self.model.save(word2vec_name)
        np.save(embedding_name, self.embedding_matrix)
        with open(tokenizer_name, 'wb') as file:
            pickle.dump(self.tokenizer, file)

    @staticmethod
    def load(word2vec_name='word2vec_model.bin', embedding_name='embedding_matrix.npy', tokenizer_name='tokenizer.pkl'):
        instance = LanguageModel()
        instance.model = Word2Vec.load(word2vec_name)
        instance.embedding_matrix = np.load(embedding_name)
        with open(tokenizer_name, 'rb') as file:
            instance.tokenizer = pickle.load(file)

        return instance

    def preprocess(self, reviews):
        sequences = self.tokenizer.texts_to_sequences(reviews)
        sequences_padded = pad_sequences(sequences, maxlen=self.max_sequence_length)

        return sequences_padded

    @property
    def get_max_sequence_length(self):
        return self.max_sequence_length

    @property
    def get_embedding_matrix(self):
        return self.embedding_matrix

    @property
    def get_tokenizer(self):
        return self.tokenizer
