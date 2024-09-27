import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.layers import LSTM, Dense, Embedding, InputLayer
from tensorflow.keras.models import Sequential, load_model


class BestModelEverLOL:
    def __init__(self, embedding_matrix=None, max_sequence_length=None, batch_size=None):
        self.batch_size = batch_size
        self.model_name = ""

        self.batch_size = batch_size
        self.model_name = ""

        if embedding_matrix is not None and max_sequence_length is not None:
            self.max_sequence_length = max_sequence_length
            self.vocab_size, self.embedding_dim = embedding_matrix.shape
            self.model = Sequential()
            # Define input data shape
            self.model.add(InputLayer(batch_input_shape=(batch_size, self.max_sequence_length,)))
            # Vectorization
            self.model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, embeddings_initializer=Constant(embedding_matrix), trainable=False))
            # LSTMs
            self.model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.2))

            # Denses
            self.model.add(Dense(1, activation='sigmoid'))

            self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['binary_accuracy'])
        else:
            self.model = None

    def train(self, X_train, y_train, X_val, y_val, num_epochs):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=self.batch_size)

    def save(self, filename):
        self.model_name = filename
        self.model.save(filename)

    @staticmethod
    def load(filename='lstm_model.keras'):
        instance = BestModelEverLOL()
        instance.model = load_model(filename)
        instance.model_name = filename
        instance.batch_size = instance.model.layers[0].input.shape[0]

        return instance

    @property
    def name(self):
        return self.model_name

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        return loss, accuracy

    def predict(self, X_test, batch_size=None, verbose=0):
        if batch_size is None:
            batch_size = self.batch_size
        return self.model.predict(X_test, batch_size=batch_size, verbose=verbose)

    def show_confision_matrix(self, X_test, y_test):
        y_pred_probs = self.predict(X_test, batch_size=self.batch_size)
        y_pred = (y_pred_probs > 0.5).astype(int)

        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        plt.figure(constrained_layout=True)
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def show_roc_curve(self, X_test, y_test):
        y_pred_probs = self.predict(X_test, batch_size=self.batch_size)

        y_pred = (y_pred_probs > 0.5).astype(int)
        y_pred_probs = y_pred_probs.ravel()

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

        # Отчет по классификации
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        print('---------------------------------------------------------------------------------------')
        print()
