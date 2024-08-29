import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
from gensim.models import Word2Vec

# Настройки
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 1
EPOCHS = 5
LEARNING_RATE = 0.001


# Класс Dataset для загрузки данных
class ConversationDataset(Dataset):
    def __init__(self, tokenized_texts, labels, word2idx):
        self.tokenized_texts = tokenized_texts
        self.labels = labels
        self.word2idx = word2idx

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        text = self.tokenized_texts[idx]
        label = self.labels[idx]
        indices = [self.word2idx.get(word, 0) for word in text]  # Предотвращаем использование неизвестных слов
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)


# Определение модели
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, embedding_matrix, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Используем предобученные эмбеддинги
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)  # Размерность [batch_size, seq_len, embedding_dim]
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.fc(lstm_out[:, -1, :])  # Выход только для последнего элемента последовательности
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))


# Загрузка данных
positive_file = 'TrainingDataPositive.txt'
negative_file = 'TrainingDataNegative.txt'

with open(positive_file, 'r') as f:
    positive_talks = f.read().splitlines()
with open(negative_file, 'r') as f:
    negative_talks = f.read().splitlines()

texts = positive_talks + negative_talks
labels = [1] * len(positive_talks) + [0] * len(negative_talks)

# Разделение на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

# Токенизация данных
train_tokenized_texts = [text.split() for text in train_texts]
test_tokenized_texts = [text.split() for text in test_texts]

# Обучение модели Word2Vec
w2v_model = Word2Vec(sentences=train_tokenized_texts, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)

# Преобразование в индексированные вектора
word2idx = {word: idx for idx, word in enumerate(w2v_model.wv.index_to_key)}
embedding_matrix = torch.tensor(w2v_model.wv.vectors, dtype=torch.float32)

# Создание датасетов и загрузчиков данных
train_dataset = ConversationDataset(train_tokenized_texts, train_labels, word2idx)
test_dataset = ConversationDataset(test_tokenized_texts, test_labels, word2idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Инициализация модели
model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(word2idx), output_size=1, embedding_matrix=embedding_matrix)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Функция обучения
def train_model(model, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        hidden = model.init_hidden(BATCH_SIZE)
        for texts, labels in train_loader:
            optimizer.zero_grad()
            texts = pad_sequence(texts, batch_first=True)  # Выравнивание последовательностей
            output, hidden = model(texts, hidden)
            loss = loss_function(output.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


# Функция тестирования
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE)
        for texts, labels in test_loader:
            texts = pad_sequence(texts, batch_first=True)  # Выравнивание последовательностей
            output, hidden = model(texts, hidden)
            predicted = torch.round(torch.sigmoid(output.squeeze()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {correct / total:.4f}')


# Функция для передачи текста по частям
def process_conversation(model, conversation_parts, word2idx, hidden=None):
    model.eval()
    if hidden is None:
        hidden = model.init_hidden(BATCH_SIZE)

    for part in conversation_parts:
        tokenized_part = torch.tensor([word2idx.get(word, 0) for word in part.split()], dtype=torch.long).unsqueeze(0)
        tokenized_part = pad_sequence([tokenized_part], batch_first=True)
        output, hidden = model(tokenized_part, hidden)

    return output, hidden


# Обучение модели
train_model(model, train_loader, EPOCHS)

# Тестирование модели
test_model(model, test_loader)

# Пример использования процесса обработки разговора частями
conversation = [
    "Hello, how are you?",
    "I am fine, thank you.",
    "Can you provide your account details?",
    "Why do you need them?"
]

# Процесс обработки диалога
hidden_state = None
for part in conversation:
    output, hidden_state = process_conversation(model, [part], word2idx, hidden_state)
