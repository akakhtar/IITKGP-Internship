import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# List of speaker file names (without extensions)
speakers = ['joey', 'ross', 'monica', 'chandler', 'rachel', 'phoebe']

# Initialize dictionaries to store results and data points
results = {}
confusion_matrices = {}
data_points = {'Speaker': [], 'Train': [], 'Test': [], 'Total': []}

for speaker in speakers:
    train = pd.read_csv(f'../Self Report Features/Final Speakers Data/{speaker}_train.csv')
    test = pd.read_csv(f'../Self Report Features/Final Speakers Data/{speaker}_test.csv')

    # Collect data point information
    train_size = train.shape[0]
    test_size = test.shape[0]
    total_size = train_size + test_size

    data_points['Speaker'].append(speaker.capitalize())
    data_points['Train'].append(train_size)
    data_points['Test'].append(test_size)
    data_points['Total'].append(total_size)

    print(f"\nProcessing Speaker: {speaker.capitalize()}")
    print(f"Shape of Train Data Set : {train.shape}")
    print(f"Shape of Test Data Set : {test.shape}")
    print(f"Total Data Points for {speaker.capitalize()}: {total_size}")

    X_train = train[['Influence_0', 'Influence_1', 'Sequence_Length', 'audio_feature', 'facial_feature']]
    y_train = train['valence']

    X_test = test[['Influence_0', 'Influence_1', 'Sequence_Length', 'audio_feature', 'facial_feature']]
    y_test = test['valence']

    model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Save results
    results[speaker] = classification_report(y_test, y_pred, output_dict=True)
    confusion_matrices[speaker] = confusion_matrix(y_test, y_pred)

    print(f"Classification Report for {speaker.capitalize()}:\n{classification_report(y_test, y_pred)}")

# Convert data points dictionary to DataFrame for easier visualization
data_points_df = pd.DataFrame(data_points)

# Plotting comparison of classification reports
plt.figure(figsize=(12, 8))

for metric in ['precision', 'recall', 'f1-score']:
    metric_values = [results[speaker]['weighted avg'][metric] for speaker in speakers]
    plt.plot(speakers, metric_values, marker='o', label=metric.capitalize())

# Annotate total data points for each speaker
for idx, speaker in enumerate(speakers):
    total_dp = data_points_df.loc[idx, 'Total']
    plt.text(idx, max(metric_values) + 0.02, f'Total: {total_dp}', ha='center', fontsize=9, color='black')

plt.title('Comparison of Precision, Recall, and F1-Score Across Speakers with Data Points')
plt.xlabel('Speakers')
plt.ylabel('Score')
plt.ylim(0.7, 1)  # Set y-axis limits for better visibility
plt.legend()
plt.show()

