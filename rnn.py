import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re


# Функция для предобработки текста
def preprocess_text(text):
    # Приведение текста к нижнему регистру
    text = text.lower()
    # Удаление всех символов кроме букв и цифр
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Токенизация и удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Чтение данных из файлов и предварительная обработка
def load_data(positive_file, negative_file):
    with open(positive_file, 'r') as f:
        positive_reviews = [preprocess_text(line.strip()) for line in f]

    with open(negative_file, 'r') as f:
        negative_reviews = [preprocess_text(line.strip()) for line in f]

    # Объединение положительных и отрицательных отзывов и создание меток
    reviews = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

    return reviews, labels


class RealtimeFraudDetectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RealtimeFraudDetectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def forward(self, x):
        # Прямое распространение через LSTM с сохранением скрытого состояния
        out, self.hidden = self.lstm(x, self.hidden)
        out = self.fc(out[:, -1, :])  # Используем последний временной шаг
        return out

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.hidden = (h0, c0)

    def reset_hidden(self):
        self.hidden = None


# Параметры модели
input_size = 10  # Количество признаков (должно совпадать с размерностью вектора)
hidden_size = 50
output_size = 2
num_layers = 2
batch_size = 32
epochs = 5
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RealtimeFraudDetectionLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Векторизация текста
def vectorize_text(reviews):
    vectorizer = CountVectorizer(max_features=input_size)
    vectors = vectorizer.fit_transform(reviews).toarray()
    return vectors


# Подготовка данных
positive_file = 'TrainingDataPositive.txt'
negative_file = 'TrainingDataNegative.txt'

reviews, labels = load_data(positive_file, negative_file)

# Векторизация текстов
X = vectorize_text(reviews)
y = np.array(labels)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Преобразование данных в тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_data = torch.utils.data.TensorDataset(X_train, y_train)
test_data = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Обучение модели
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Преобразование входов в 3D-тензор (batch_size, sequence_length, input_size)
        inputs = inputs.unsqueeze(1)  # Добавляем измерение последовательности

        model.init_hidden(inputs.size(0), device)

        # Прямое распространение
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Обратное распространение
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Оценка модели на тестовых данных
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Преобразование входов в 3D-тензор (batch_size, sequence_length, input_size)
        inputs = inputs.unsqueeze(1)  # Добавляем измерение последовательности

        model.init_hidden(inputs.size(0), device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
