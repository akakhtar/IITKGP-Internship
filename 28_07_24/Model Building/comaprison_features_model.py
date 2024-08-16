import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the data
train = pd.read_csv('../Self Report Features/Final Speakers Data/ross_train.csv')
test = pd.read_csv('../Self Report Features/Final Speakers Data/ross_test.csv')

# Print dataset shapes
print(f"Shape of Train Data Set : {train.shape}")
print(f"Shape of Test Data Set : {test.shape}")

# Print counts of valence and arousal
print("\nCount of valence :")
print(f"Train: \n{train['valence'].value_counts()}")
print(f"Test : \n{test['valence'].value_counts()}")
print("\nCount of arousal :")
print(f"Train: \n{train['arousal'].value_counts()}")
print(f"Test : \n{test['arousal'].value_counts()}")

# Define feature sets
feature_sets = {
    "Self_Report + Facial": ['Influence_0', 'Influence_1', 'Sequence_Length', 'facial_feature'],
    "Self_Report + Audio": ['Influence_0', 'Influence_1', 'Sequence_Length', 'audio_feature'],
    "Self_Report + Lexical": ['Influence_0', 'Influence_1', 'Sequence_Length', 'lexical_feature'],
    "All Features": ['Influence_0', 'Influence_1', 'Sequence_Length', 'lexical_feature', 'audio_feature',
                     'facial_feature']
}

results = {}

# Loop through each feature set and evaluate
for feature_set_name, features in feature_sets.items():
    X_train = train[features]
    y_train = train['valence']
    X_test = test[features]
    y_test = test['valence']

    # Train the model
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store the results
    results[feature_set_name] = {
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {feature_set_name}')
    plt.show()

# Data Points Information
num_train_points = train.shape[0]
num_test_points = test.shape[0]

# Plot comparison of Precision, Recall, and F1-Score
metrics = ['precision', 'recall', 'f1-score']

for metric in metrics:
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [results[name][metric] for name in results], color='skyblue')
    plt.xlabel('Feature Set')
    plt.ylabel(f'Weighted {metric.capitalize()}')
    plt.title(f'Comparison of {metric.capitalize()} Across Feature Sets\n'
              f'Train Data Points: {num_train_points}, Test Data Points: {num_test_points}')
    plt.show()
