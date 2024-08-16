import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the data
train = pd.read_csv('../Self Report Features/Final Speakers Data/ross_train.csv')
test = pd.read_csv('../Self Report Features/Final Speakers Data/ross_test.csv')

# Define feature sets
feature_sets = {
    "self" : ['Influence_0','Influence_1','Sequence_Length'],
    "self,facial": ['Influence_0', 'Influence_1', 'Sequence_Length', 'facial_feature'],
    "self,audio": ['Influence_0', 'Influence_1', 'Sequence_Length', 'audio_feature'],
    "self,audio,facial": ['Influence_0', 'Influence_1', 'Sequence_Length','facial_feature', 'audio_feature'],
    "self,lexical": ['Influence_0', 'Influence_1', 'Sequence_Length', 'lexical_feature'],
    "self,facial,lexical" : ['Influence_0', 'Influence_1', 'Sequence_Length', 'facial_feature','lexical_feature'],
    "self,audio,lexical" : ['Influence_0', 'Influence_1', 'Sequence_Length', 'audio_feature','lexical_feature'],
    "self,facial,audio,lexical": ['Influence_0', 'Influence_1', 'Sequence_Length', 'lexical_feature', 'audio_feature',
                     'facial_feature']
}

results = {'Feature Set': [], 'Class': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

# Loop through each feature set and evaluate
for feature_set_name, features in feature_sets.items():
    X_train = train[features]
    y_train = train['valence']
    X_test = test[features]
    y_test = test['valence']

    # Train the model
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=42)
    # model = SVC(random_state=42)
    # model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store the results for both classes 0 and 1
    for cls in ['0', '1']:
        results['Feature Set'].append(feature_set_name)
        results['Class'].append(f"Class {cls}")
        results['Precision'].append(report[cls]['precision'])
        results['Recall'].append(report[cls]['recall'])
        results['F1-Score'].append(report[cls]['f1-score'])

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plotting Precision and Recall
plt.figure(figsize=(10, 6))

# Precision
plt.plot(results_df[results_df['Class'] == "Class 0"]['Feature Set'],
         results_df[results_df['Class'] == "Class 0"]['Precision'],
         marker='o', linestyle='-', color='orange', label='Precision Class 0')

plt.plot(results_df[results_df['Class'] == "Class 1"]['Feature Set'],
         results_df[results_df['Class'] == "Class 1"]['Precision'],
         marker='o', linestyle='--', color='orange', label='Precision Class 1')

# Recall
plt.plot(results_df[results_df['Class'] == "Class 0"]['Feature Set'],
         results_df[results_df['Class'] == "Class 0"]['Recall'],
         marker='o', linestyle='-', color='green', label='Recall Class 0')

plt.plot(results_df[results_df['Class'] == "Class 1"]['Feature Set'],
         results_df[results_df['Class'] == "Class 1"]['Recall'],
         marker='o', linestyle='--', color='green', label='Recall Class 1')

# Customize the plot
plt.xlabel('Feature Sets')
plt.ylabel('Score')
plt.title('Precision and Recall Across Different Feature Sets (Classes 0 and 1)')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting F1-Score
plt.figure(figsize=(10, 6))

# F1-Score
plt.plot(results_df[results_df['Class'] == "Class 0"]['Feature Set'],
         results_df[results_df['Class'] == "Class 0"]['F1-Score'],
         marker='o', linestyle='-', color='red', label='F1-Score Class 0')

plt.plot(results_df[results_df['Class'] == "Class 1"]['Feature Set'],
         results_df[results_df['Class'] == "Class 1"]['F1-Score'],
         marker='o', linestyle='--', color='red', label='F1-Score Class 1')

# Customize the plot
plt.xlabel('Feature Sets')
plt.ylabel('F1-Score')
plt.title('F1-Score Across Different Feature Sets (Classes 0 and 1)')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
