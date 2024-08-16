import re
import time

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA


# Load pre-trained BERT model and tokenizer
startTimeModel = time.time()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
endTimeModel = time.time()
print(f"Time taken to load pre-trained BERT model: {endTimeModel-startTimeModel}")

#Loading of data set
train = pd.read_csv('../Self Report Features/Final Speakers Data/ross_train.csv')
test = pd.read_csv('../Self Report Features/Final Speakers Data/ross_test.csv')


def preprocess_text(text):
    # Remove character encoding artifacts
    text = re.sub(r'Â', '', text)
    return text

# Function to get features from text
def get_bert_features(text):
    text = preprocess_text(text)
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Get hidden states from BERT model
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state

    # Get the embeddings for the [CLS] token (first token)
    cls_embeddings = hidden_states[:, 0, :]

    return cls_embeddings.squeeze().numpy()

def get_combined_features(data):
    features = []
    for index, row in data.iterrows():
        features.append(get_bert_features(row['Utterance']))
    return features

train_features = get_combined_features(train)
test_features = get_combined_features(test)

scaler  = StandardScaler()
scaled_features = scaler.fit_transform(train_features)
kpca = KernelPCA(n_components=1,kernel='rbf',eigen_solver='arpack')
train_features_reduced = kpca.fit_transform(scaled_features)

train["lexical_feature"] = train_features_reduced

scaled_test_features = scaler.transform(test_features)
test_features_reduced = kpca.transform(scaled_test_features)
test["lexical_feature"] = test_features_reduced

train.to_csv('../Self Report Features/Final Speakers Data/ross_train.csv', index=False)
test.to_csv('../Self Report Features/Final Speakers Data/ross_test.csv', index=False)




