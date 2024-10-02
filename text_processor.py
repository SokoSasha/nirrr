import pickle
import string
import time

import contractions
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class LanguageModel:
    def __init__(self, vector_size=300, window=10, min_count=5, workers=4):
        self.__model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        self.__tokenizer = Tokenizer()
        self.__embedding_matrix = None
        self.__stop_words = set(stopwords.words('english'))  # Список стоп-слов
        self.__stop_words.discard('no')
        self.__stop_words.discard('not')
        self.__lemmatizer = WordNetLemmatizer()  # Лемматизатор

    def train(self, texts: list[str], epochs=5, update=False, check=False):
        print("Training model... ", end="")
        start_time = time.perf_counter()

        all_tokens = self.text_to_txt_tokens(texts)
        self.__tokenizer.fit_on_texts(all_tokens)

        self.__model.build_vocab(all_tokens, update=update)
        history = self.__model.train(all_tokens, total_examples=len(texts), epochs=epochs)

        elapsed_time = time.perf_counter() - start_time
        print(f"done in {elapsed_time:.4f} seconds")

        print("Generating embedding matrix...", end="")
        start_time = time.perf_counter()

        word_index = self.__tokenizer.word_index
        self.__embedding_matrix = np.zeros((len(word_index) + 1, self.__model.vector_size))
        for word, i in word_index.items():
            if word in self.__model.wv:
                self.__embedding_matrix[i] = self.__model.wv[word]

        elapsed_time = time.perf_counter() - start_time
        print(f"done in {elapsed_time:.4f} seconds")

        if check:
            most_similar_words = self.__model.wv.most_similar('good', topn=5)
            print(f"Most similar words to 'good': {most_similar_words}")

            most_similar_words = self.__model.wv.most_similar('bad', topn=5)
            print(f"Most similar words to 'bad': {most_similar_words}")

        return history

    def save(self, word2vec_name='word2vec_model.bin', embedding_name='embedding_matrix.npy',
             tokenizer_name='tokenizer.pkl'):
        print("Saving model... ", end="")
        start_time = time.perf_counter()

        self.__model.save(word2vec_name)
        np.save(embedding_name, self.__embedding_matrix)
        with open(tokenizer_name, 'wb') as file:
            pickle.dump(self.__tokenizer, file)

        elapsed_time = time.perf_counter() - start_time
        print(f"done in {elapsed_time:.4f} seconds")

    @staticmethod
    def load(word2vec_name='word2vec_model.bin', embedding_name='embedding_matrix.npy', tokenizer_name='tokenizer.pkl'):
        print("Loading model... ", end="")
        start_time = time.perf_counter()

        instance = LanguageModel()
        instance.__model = Word2Vec.load(word2vec_name)
        instance.__embedding_matrix = np.load(embedding_name)
        with open(tokenizer_name, 'rb') as file:
            instance.__tokenizer = pickle.load(file)

        elapsed_time = time.perf_counter() - start_time
        print(f"done in {elapsed_time:.4f} seconds")
        return instance

    def text_to_txt_tokens(self, texts):
        all_tokens = []
        for text in texts:
            expanded_review = contractions.fix(text)
            tokens = [
                self.__lemmatizer.lemmatize(word)
                for word in word_tokenize(expanded_review.lower())
                if word not in self.__stop_words and word not in string.punctuation
            ]
            all_tokens.append(tokens)

        return all_tokens

    def texts_to_sequences(self, texts):
        sequences = self.__tokenizer.texts_to_sequences(self.text_to_txt_tokens(texts))
        return sequences

    @staticmethod
    def pad_sequences(sequences, maxlen=200):
        return pad_sequences(sequences, maxlen=maxlen)

    @property
    def get_embedding_matrix(self):
        return self.__embedding_matrix
