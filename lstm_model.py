import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.src.initializers import Constant
from keras.src.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, InputLayer
from tensorflow.keras.models import Sequential, load_model


class BestModelEverLOL:
    def __init__(self, embedding_matrix, max_sequence_length, batch_size):
        self.batch_size = batch_size
        self.model_name = ""

        vocab_size, embedding_dim = embedding_matrix.shape
        self.model = Sequential()
        # Define input data shape
        self.model.add(InputLayer(batch_input_shape=(batch_size, max_sequence_length,)))
        # Vectorization
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer=Constant(embedding_matrix), trainable=False))
        # LSTMs
        self.model.add(LSTM(64, stateful=True, dropout=0.05, recurrent_dropout=0.2))
        # Denses
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['binary_accuracy'])

        self.model_description = self._gen_model_description()

    def _gen_model_description(self):
        model_desc = ""
        for layer in self.model.layers:
            if isinstance(layer, Bidirectional):
                model_desc += f"{layer.name}({layer.forward_layer.name}) [{layer.input.shape}]\n"
            elif isinstance(layer, Dense):
                model_desc += f"{layer.name}({layer.activation.__name__}) [{layer.input.shape}]\n"
            else:
                model_desc += f"{layer.name} [{layer.input.shape}]\n"

        return model_desc

    def train(self, X_train, y_train, X_val, y_val, num_epochs):
        for ep in range(num_epochs):
            print(f"Epoch {ep+1}/{num_epochs}")
            self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=self.batch_size)
            self.reset_state()

    def save(self, filename):
        self.model_name = filename
        self.model.save(filename)

    @staticmethod
    def load(filename='lstm_model.keras'):
        tmp_model = load_model(filename)
        batch_size = tmp_model.layers[0].input.shape[0]
        max_sequence_length = tmp_model.layers[0].input.shape[1]
        embedding_matrix = tmp_model.layers[0].embeddings_initializer.value
        instance = BestModelEverLOL(embedding_matrix, max_sequence_length, batch_size)

        for i in range(len(tmp_model.layers)):
            instance.model.layers[i].set_weights(tmp_model.layers[i].get_weights())

        return instance

    @property
    def name(self):
        return self.model_name

    @property
    def description(self):
        return self.model_description

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        return loss, accuracy

    def predict(self, X_test, batch_size=None, verbose=0):
        if batch_size is None:
            batch_size = self.batch_size
        return self.model.predict(X_test, batch_size=batch_size, verbose=verbose)

    def reset_state(self, verbose=0):
        if verbose:
            print("Resetting model...")

        for layer in self.model.layers:
            try:
                layer.reset_states()
            except AttributeError:
                continue

    def show_confision_matrix(self, X_test, y_test, show_description=True):
        y_pred_probs = self.predict(X_test, batch_size=self.batch_size)
        y_pred = (y_pred_probs > 0.5).astype(int)

        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        plt.figure(constrained_layout=True)
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        if show_description:
            plt.suptitle(self.model_description)
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
