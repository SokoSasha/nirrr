import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, InputLayer
from tensorflow.keras.models import Sequential

from text_processing import *

BATCH_SIZE = 32
NUM_EPOCHS = 3


def show_data(y_pred_probs, y_test, model_desc):
    y_pred = (y_pred_probs > 0.5).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum()

    plt.figure(constrained_layout=True)
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='YlGn', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.suptitle(model_desc)
    plt.show()

    # Отчет по классификации
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    print('---------------------------------------------------------------------------------------')
    print()


def get_model_desc(model):
    model_desc = ""
    for layer in model.layers:
        if isinstance(layer, Bidirectional):
            model_desc += f"{layer.name}({layer.forward_layer.name}) [{layer.input.shape}]\n"
        elif isinstance(layer, Dense):
            model_desc += f"{layer.name}({layer.activation.__name__}) [{layer.input.shape}]\n"
        else:
            model_desc += f"{layer.name} [{layer.input.shape}]\n"

    print(f"[LOG]\n{model_desc}")
    return model_desc


# Создание модели LSTM
def create_lstm_model(embedding_matrix, max_sequence_length, batch_size, activation_f):
    vocab_size, embedding_dim = embedding_matrix.shape
    model = Sequential()

    # Define input data shape
    model.add(InputLayer(batch_input_shape=(batch_size, max_sequence_length,)))
    #
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=False))
    model.add(Bidirectional(LSTM(32, stateful=True, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(32, stateful=True, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)))

    # Добавление слоя Dense
    model.add(Dense(1, activation=activation_f))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


def main():
    positive_file = "TrainingDataPositive.txt"
    negative_file = "TrainingDataNegative.txt"
    test_filepath = 'TestReviews.csv'

    X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix, max_sequence_length = get_data(positive_file, negative_file, test_filepath, BATCH_SIZE)

    for activ in ['sigmoid', 'softplus', 'softsign', 'tanh']:
        # Создание и обучение модели LSTM
        model = create_lstm_model(embedding_matrix, max_sequence_length, BATCH_SIZE, activ)

        model_desc = get_model_desc(model)

        checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), shuffle=False, callbacks=[checkpoint])

        # Оценка модели
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
        print(f"Test Accuracy: {accuracy * 100:.2f}")

        y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE)
        show_data(y_pred_probs, y_test, model_desc)


if __name__ == "__main__":
    main()
