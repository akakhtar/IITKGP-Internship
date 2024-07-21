from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

train = pd.read_csv('/content/drive/My Drive/CMED/ross/ross_train.csv')
test = pd.read_csv('/content/drive/My Drive/CMED/ross/ross_test.csv')

# Prepare the features and labels
X = train[["Influence_0", "Influence_1", "Sequence_Length", "facial_feature", "audio_feature"]]
y = train["valence"]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


# Evaluate the model on the test set
X_test = test[["Influence_0", "Influence_1", "Sequence_Length", "facial_feature", "audio_feature"]]
y_test = test["valence"]

y_test_pred = model.predict(X_test)
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Cross-Validation
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("Cross-Validation Accuracy Scores:", scores)
print("Average Cross-Validation Accuracy:", scores.mean())

