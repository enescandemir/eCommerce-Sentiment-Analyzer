import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('processed_comments.csv')

labels = df['Comment_Context_Ticket'].dropna()

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)
print(f"Farklı sınıf sayısı (num_classes): {num_classes}")
