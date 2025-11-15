import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import sys

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

def load_model(path, model_class, vocab_size, device='cuda'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    num_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
    hyperparameters = {k: v for k, v in checkpoint['hyperparameters'].items() if k in model_class.__init__.__code__.co_varnames and k != 'num_classes'}
    model = model_class(vocab_size=vocab_size, num_classes=num_classes, **hyperparameters)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def tokenize_texts(texts, tokenizer, max_length=128):
    tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length) for text in texts]
    return pad_sequence([torch.tensor(t) for t in tokenized_texts], batch_first=True)

def fill_missing_tickets(csv_path, model_path, output_csv):
    df = pd.read_csv(csv_path)
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    texts = df['Comment_Context'].tolist()
    label_encoder = LabelEncoder().fit(df['Comment_Context_Ticket'].dropna())

    vocab_size = tokenizer.vocab_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, TextCNN, vocab_size, device)
    tokenized_texts = tokenize_texts(texts, tokenizer)

    with torch.no_grad():
        for i in range(len(texts)):
            if pd.isnull(df.at[i, 'Comment_Context_Ticket']):
                input_tensor = tokenized_texts[i].unsqueeze(0).to(device)
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                df.at[i, 'Comment_Context_Ticket'] = predicted.cpu().item()

    df.to_csv(output_csv, index=False)
    print(f"Eksik etiketler dolduruldu ve buraya kaydedildi : {output_csv}")

if __name__ == "__main__":
    csv_path = sys.argv[1]
    output_csv = sys.argv[2]
    model_path = sys.argv[3]
    fill_missing_tickets(csv_path, model_path, output_csv)
