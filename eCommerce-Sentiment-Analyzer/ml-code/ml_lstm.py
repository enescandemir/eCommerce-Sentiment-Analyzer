import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np

df = pd.read_csv('processed_comments.csv')
df = df[['Comment_Context', 'Comment_Context_Ticket']].dropna()

texts = df['Comment_Context'].tolist()
labels = df['Comment_Context_Ticket'].tolist()

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)

tokenized_texts = [tokenize_text(text) for text in texts]

padded_texts = pad_sequence([torch.tensor(t) for t in tokenized_texts], batch_first=True)
sequence_lengths = [len(t) for t in tokenized_texts]

X_train, X_test, y_train, y_test, train_lengths, test_lengths = train_test_split(
    padded_texts, labels, sequence_lengths, test_size=0.2, random_state=42
)

class TextDataset(Dataset):
    def __init__(self, texts, labels, lengths):
        self.texts = texts
        self.labels = torch.tensor(labels)
        self.lengths = torch.tensor(lengths)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.lengths[idx]

train_dataset = TextDataset(X_train, y_train, train_lengths)
test_dataset = TextDataset(X_test, y_test, test_lengths)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        idx = (lengths - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2))
        idx = idx.unsqueeze(1).to(lstm_out.device)
        last_outputs = lstm_out.gather(1, idx).squeeze(1)

        out = self.dropout(last_outputs)
        out = self.fc(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = tokenizer.vocab_size
num_classes = len(label_encoder.classes_)

hidden_sizes = [128, 256]
embed_sizes = [64, 128]
learning_rates = [0.001, 0.0005]

best_model = None
best_accuracy = 0
best_params = {}

for hidden_size in hidden_sizes:
    for embed_size in embed_sizes:
        for lr in learning_rates:
            print(f"Test edilen parametreler: hidden_size={hidden_size}, embed_size={embed_size}, lr={lr}")

            model = TextLSTM(vocab_size, embed_size, hidden_size, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            num_epochs = 10
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                correct = 0
                total = 0

                for texts, labels, lengths in tqdm(train_loader):
                    texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)

                    outputs = model(texts, lengths)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                train_acc = 100 * correct / total
                print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.2f}%")

            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for texts, labels, lengths in test_loader:
                    texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
                    outputs = model(texts, lengths)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validasyon Doğruluğu: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_params = {
                    'hidden_size': hidden_size,
                    'embed_size': embed_size,
                    'learning_rate': lr
                }

torch.save({
    'model_state_dict': best_model.state_dict(),
    'hyperparameters': best_params
}, 'best_text_lstm_model_with_params.pth')

print("En iyi parametreler:", best_params)
print(f"En iyi model şu oranla kaydedildi: {best_accuracy:.4f}")

best_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels, lengths in test_loader:
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        outputs = best_model(texts, lengths)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)

f1 = f1_score(all_labels, all_preds, average='weighted')

conf_matrix = confusion_matrix(all_labels, all_preds)

true_positives = np.diag(conf_matrix)
false_positives = np.sum(conf_matrix, axis=0) - true_positives
false_negatives = np.sum(conf_matrix, axis=1) - true_positives
true_negatives = np.sum(conf_matrix) - (false_positives + false_negatives + true_positives)

sensitivity = true_positives / (true_positives + false_negatives + 1e-10)
specificity = true_negatives / (true_negatives + false_positives + 1e-10)

mean_sensitivity = np.mean(sensitivity)
mean_specificity = np.mean(specificity)

print("En iyi hiperparametre kombinasyonu:")
for param in ['embed_size', 'hidden_size', 'learning_rate']:
    print(f"{param}: {best_params[param]}")

print(f"\nTest Doğruluğu (Accuracy): {accuracy:.4f}")
print(f"F1 Skoru: {f1:.4f}")
print(f"Duyarlılık (Sensitivity): {mean_sensitivity:.4f}")
print(f"Özgüllük (Specificity): {mean_specificity:.4f}")
print("\nKonfüzyon Matrisi:")
print(conf_matrix)
