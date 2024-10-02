import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import LSTM, Dense, Embedding, InputLayer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


def tf_precision(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true, 'float32') * tf.cast(y_pred, 'float32'), axis=0)
    precision = tp / (tf.reduce_sum(tf.cast(y_pred, 'float'), axis=0) + tf.keras.backend.epsilon())
    return precision


def tf_recall(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true, 'float32') * tf.cast(y_pred, 'float32'), axis=0)
    recall = tp / (tf.reduce_sum(tf.cast(y_true, 'float'), axis=0) + tf.keras.backend.epsilon())
    return recall


class BestModelEverLOL:
    def __init__(self, embedding_matrix=None, max_sequence_length=None, batch_size=None):
        self.__batch_size = batch_size
        self.__model_name = ""

        if embedding_matrix is not None and max_sequence_length is not None:
            self.__max_sequence_length = max_sequence_length
            vocab_size, embedding_dim = embedding_matrix.shape
            self.__model = Sequential()
            # Define input data shape
            self.__model.add(InputLayer(batch_input_shape=(batch_size, self.__max_sequence_length,)))
            # Vectorization
            self.__model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                       embeddings_initializer=Constant(embedding_matrix), trainable=False))
            # LSTMs
            # self.__model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.2, stateful=True))
            self.__model.add(LSTM(32, return_sequences=True, stateful=True, dropout=0.5, recurrent_dropout=0.5))
            self.__model.add(LSTM(16, stateful=True))

            # Denses
            self.__model.add(Dense(1, activation='sigmoid'))

            self.__model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001),
                                 metrics=['binary_accuracy', tf_precision, tf_recall])
        else:
            self.__model = None

    def train(self, X_train, y_train, X_val, y_val, num_epochs, class_weight):
        # self.__model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=self.__batch_size)
        for i in range(num_epochs):
            print(f"Epoch {(i + 1)}/{num_epochs}")
            self.__model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=self.__batch_size,
                             class_weight=class_weight)
            self.reset()

    def save(self, filename):
        self.__model_name = filename
        self.__model.save(filename)

    def reset(self):
        for layer in self.__model.layers:
            try:
                layer.reset_states()
            except:
                continue

    @staticmethod
    def load(filename='lstm_model.keras'):
        instance = BestModelEverLOL()
        instance.__model = load_model(filename)
        instance.__model_name = filename
        instance.__batch_size = instance.__model.layers[0].input.shape[0]
        instance.__max_sequence_length = instance.__model.layers[0].input.shape[1]

        return instance

    @property
    def name(self):
        return self.__model_name

    @property
    def get_max_sequence_length(self):
        return self.__max_sequence_length

    def summary(self):
        self.__model.summary(expand_nested=True, show_trainable=True)

    def evaluate(self, X_test, y_test):
        res = self.__model.evaluate(X_test, y_test, batch_size=self.__batch_size)
        return res

    def predict(self, X_test, batch_size=None, verbose=False):
        if batch_size is None:
            batch_size = self.__batch_size
        return self.__model.predict(X_test, batch_size=batch_size, verbose=verbose)

    def show_confision_matrix(self, y_pred, y_test, title='Confusion matrix'):
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_normalized = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        plt.figure(constrained_layout=True)
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Greens', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.show()

    def show_roc_curve(self, y_pred, y_test):
        y_pred_probs = y_pred.ravel()

        # Вычисление ROC-кривой
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        # Построение ROC-кривой
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def print_metrics(self, y_pred, y_test):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Вывод метрик
        print('-----------------')
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1-score:  {f1 * 100:.2f}%")
        print('-----------------')
