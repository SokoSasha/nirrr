import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


# Объявляем класс Dataset, который хранит тексты и метки
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Определяем модель LSTM
class RealtimeFraudDetectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RealtimeFraudDetectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def forward(self, x):
        out, self.hidden = self.lstm(x.unsqueeze(1), self.hidden)  # Добавляем размер последовательности
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.hidden = (h0, c0)

    def reset_hidden(self):
        self.hidden = None


# Предварительная обработка и векторизация текста
def preprocess_and_vectorize(text, vectorizer):
    text = re.sub(r'[^\w\s]', '', text.lower())
    vectorized = vectorizer.transform([text]).toarray()
    vectorized = torch.tensor(vectorized, dtype=torch.float32)

    # Если размер вектора меньше input_size, добавляем padding
    if vectorized.size(1) < input_size:
        padding = torch.zeros(1, input_size - vectorized.size(1))
        vectorized = torch.cat((vectorized, padding), dim=1)

    return vectorized


# Параметры модели
input_size = 1000  # Количество признаков после векторизации
hidden_size = 50
output_size = 2
num_layers = 2
batch_size = 32
epochs = 5
learning_rate = 0.001

# Пример текстов и меток
positive_file = 'TrainingDataPositive.txt'
negative_file = 'TrainingDataNegative.txt'

with open(positive_file, 'r') as f:
    positive_reviews = f.read().splitlines()
with open(negative_file, 'r') as f:
    negative_reviews = f.read().splitlines()

texts = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

# Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Инициализация векторизатора
vectorizer = CountVectorizer(max_features=input_size)
vectorizer.fit(train_texts)

# Создание датасетов
train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Инициализация модели, оптимизатора и функции потерь
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RealtimeFraudDetectionLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(epochs):
    model.train()
    for texts, labels in train_loader:
        # Векторизация текста
        # inputs = torch.cat([preprocess_and_vectorize(text, vectorizer) for text in texts]).to(device)
        inputs = preprocess_and_vectorize(texts[0], vectorizer).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        model.init_hidden(inputs.size(0), device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Оценка модели на тестовых данных
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for texts, labels in test_loader:
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # Отображаем текст, передаваемый модели
        print(f"Input text: {texts[0]}")

        # Векторизация текста
        vectorized = preprocess_and_vectorize(texts[0], vectorizer)
        print(f"Vectorized text shape: {vectorized.shape}")  # Проверка размера вектора

        # Векторизация текста непосредственно перед передачей модели
        inputs = torch.tensor(vectorized, dtype=torch.float32).to(device)

        # Если размер вектора меньше ожидаемого input_size
        if inputs.size(-1) < input_size:
            padding = torch.zeros(1, input_size - inputs.size(-1)).to(device)
            inputs = torch.cat((inputs, padding), dim=-1)

        model.init_hidden(inputs.size(0), device)
        outputs = model(inputs)

        # Применение softmax для получения вероятностей
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        # Извлечение вероятности для негативного класса (0)
        negative_probability = probabilities[0][0].item() * 100

        # Вывод результата
        print(f"Prediction: {negative_probability:.2f}% confidence that the review is negative\n")

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
