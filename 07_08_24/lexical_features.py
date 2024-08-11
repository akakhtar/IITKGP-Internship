import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Function to get features from text
def get_bert_features(text):
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
text = "BERT is an amazing model for NLP tasks!"
features = get_bert_features(text)
print(features)
