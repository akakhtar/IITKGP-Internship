import re
import time
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

startTimeModel = time.time()
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
endTimeModel = time.time()
print(f"Time taken to load pre-trained BERT model: {endTimeModel-startTimeModel}")

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


# Example usage
anger = "AhÂ… AhÂ…Get out of here!  Uh, meeting someone? Or-or are you just here to brush up on MarionÂ’s views on evolution?"
sadness = "Listen, IÂ’m ah, IÂ’m sorry IÂ’ve been so crazy and jealous and, itÂ’s just that I like you a lot, so..."
joy = "Wow! It looks like we got a lot of good stuff."
disgust = "SheÂ’ll drive us totally crazy."
fear = "I know!  Don't switch hands, okay?"
text = [anger,sadness,joy,disgust,fear]
emotions = ["anger","sadness","joy","disgust","fear"]
features = []

startTime = time.time()
for i in range(len(text)):

    feature = get_bert_features(text[i])
    features.append(feature)
endTime = time.time()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kpca = KernelPCA(n_components=1,kernel="rbf",eigen_solver='arpack')
features_reduced = kpca.fit_transform(features_scaled)
print(features_reduced)
print(f"Time taken: {endTime-startTime}")

