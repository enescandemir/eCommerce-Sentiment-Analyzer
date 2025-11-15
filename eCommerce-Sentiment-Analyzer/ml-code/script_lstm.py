import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelEncoder
import sys

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

    model = load_model(model_path, TextLSTM, vocab_size, device)
    tokenized_texts = tokenize_texts(texts, tokenizer)

    with torch.no_grad():
        for i in range(len(texts)):
            if pd.isnull(df.at[i, 'Comment_Context_Ticket']):
                input_tensor = tokenized_texts[i].unsqueeze(0).to(device)
                length_tensor = torch.tensor([len(tokenized_texts[i])]).to(device)
                output = model(input_tensor, length_tensor)
                _, predicted = torch.max(output.data, 1)
                df.at[i, 'Comment_Context_Ticket'] = predicted.cpu().item()

    df.to_csv(output_csv, index=False)
    print(f"Eksik etiketler dolduruldu ve buraya kaydedildi : {output_csv}")

if __name__ == "__main__":
    csv_path = sys.argv[1]
    output_csv = sys.argv[2]
    model_path = sys.argv[3]
    fill_missing_tickets(csv_path, model_path, output_csv)
