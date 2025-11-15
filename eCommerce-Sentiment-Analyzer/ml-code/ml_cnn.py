import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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

X_train, X_test, y_train, y_test = train_test_split(
    padded_texts, labels, test_size=0.2, random_state=42
)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.conv1 = nn.Conv2d(1, num_filters, (3, embed_size))
        self.conv2 = nn.Conv2d(1, num_filters, (4, embed_size))
        self.conv3 = nn.Conv2d(1, num_filters, (5, embed_size))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv1_out = torch.relu(self.conv1(x)).squeeze(3)
        conv2_out = torch.relu(self.conv2(x)).squeeze(3)
        conv3_out = torch.relu(self.conv3(x)).squeeze(3)
        pool1 = torch.max(conv1_out, 2)[0]
        pool2 = torch.max(conv2_out, 2)[0]
        pool3 = torch.max(conv3_out, 2)[0]
        out = torch.cat([pool1, pool2, pool3], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

learning_rates = [0.001, 0.0005]
embed_sizes = [128, 256]
num_filters_list = [100, 150]

best_accuracy = 0
best_model = None
best_params = {}

for lr in learning_rates:
    for embed_size in embed_sizes:
        for num_filters in num_filters_list:
            print(f"Test edilen parametreler: lr={lr}, embed_size={embed_size}, num_filters={num_filters}")

            vocab_size = tokenizer.vocab_size
            num_classes = len(label_encoder.classes_)
            model = TextCNN(vocab_size, embed_size, num_classes, num_filters)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(10):
                model.train()
                epoch_loss = 0
                correct = 0
                total = 0

                with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{10}") as tepoch:
                    for texts, labels in tepoch:
                        texts, labels = texts.to(device), labels.to(device)

                        outputs = model(texts)
                        loss = criterion(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        tepoch.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for texts, labels in tqdm(test_loader, desc="Validation"):
                    texts, labels = texts.to(device), labels.to(device)
                    outputs = model(texts)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validasyon Doğruluğu: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_params = {
                    'learning_rate': lr,
                    'embed_size': embed_size,
                    'num_filters': num_filters
                }

torch.save({
    'model_state_dict': best_model.state_dict(),
    'hyperparameters': best_params
}, 'best_text_cnn_model_with_params.pth')

print("\nEn iyi model şu oranla kaydedildi:", best_accuracy)
print("En iyi parametreler:", best_params)

best_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = best_model(texts)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

true_positives = np.diag(conf_matrix)
false_positives = np.sum(conf_matrix, axis=0) - true_positives
false_negatives = np.sum(conf_matrix, axis=1) - true_positives
true_negatives = np.sum(conf_matrix) - (true_positives + false_positives + false_negatives)

sensitivity = true_positives / (true_positives + false_negatives + 1e-10)
specificity = true_negatives / (true_negatives + false_positives + 1e-10)

mean_sensitivity = np.mean(sensitivity)
mean_specificity = np.mean(specificity)

print("En iyi hiperparametre kombinasyonu:")
for param, value in best_params.items():
    print(f"{param}: {value}")

print(f"\nTest Doğruluğu (Accuracy): {accuracy:.4f}")
print(f"F1 Skoru: {f1:.4f}")
print(f"Duyarlılık (Sensitivity): {mean_sensitivity:.4f}")
print(f"Özgüllük (Specificity): {mean_specificity:.4f}")
print("\nKonfüzyon Matrisi:")
print(conf_matrix)
