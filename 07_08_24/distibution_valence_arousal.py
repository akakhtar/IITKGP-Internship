import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_path = "../28_07_24/Speakers/Train/"
dev_path = "../28_07_24/Speakers/Dev/"
test_path = "../28_07_24/Speakers/Test/"

Train = ["Ross_train.csv", "Rachel_train.csv", "Monica_train.csv", "Joey_train.csv", "Chandler_train.csv",
         "Phoebe_train.csv"]
Dev = ["Ross_dev.csv", "Rachel_dev.csv", "Monica_dev.csv", "Joey_dev.csv", "Chandler_dev.csv", "Phoebe_dev.csv"]
Test = ["Ross_test.csv", "Rachel_test.csv", "Monica_test.csv", "Joey_test.csv", "Chandler_test.csv", "Phoebe_test.csv"]
speakers = ["Ross", "Rachel", "Monica", "Joey", "Chandler", "Phoebe"]

valence_counts = []
arousal_counts = []


def find_distribution(i):
    print(f"Distribution for speaker: {speakers[i]}")
    df_train = pd.read_csv(train_path + Train[i])
    df_dev = pd.read_csv(dev_path + Dev[i])
    df_test = pd.read_csv(test_path + Test[i])
    df = pd.concat([df_train, df_dev, df_test], ignore_index=True)

    valence_count = df['valence'].value_counts().sort_index()
    arousal_count = df['arousal'].value_counts().sort_index()

    valence_counts.append(valence_count)
    arousal_counts.append(arousal_count)

    print(f"Total number of datapoints: {df.shape[0]}")
    print("Distribution of valence 0 and 1:")
    print(valence_count)
    print("Distribution of arousal 0 and 1:")
    print(arousal_count)


# Loop through all speakers
for i in range(6):
    find_distribution(i)

# Prepare data for plotting
valence_data = pd.DataFrame(valence_counts, index=speakers).fillna(0).astype(int)
arousal_data = pd.DataFrame(arousal_counts, index=speakers).fillna(0).astype(int)

# Plotting Valence
fig_valence, ax_valence = plt.subplots(figsize=(12, 5))


# Function to add counts on bars
def add_counts_on_bars(ax, data):
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


valence_data.plot(kind='bar', ax=ax_valence)
ax_valence.set_title('Valence Distribution')
ax_valence.set_ylabel('Count')
ax_valence.set_xlabel('Speakers')
ax_valence.legend(title='Valence')
add_counts_on_bars(ax_valence, valence_data)

plt.tight_layout()
plt.show()

# Plotting Arousal
fig_arousal, ax_arousal = plt.subplots(figsize=(12, 5))

arousal_data.plot(kind='bar', ax=ax_arousal)
ax_arousal.set_title('Arousal Distribution')
ax_arousal.set_ylabel('Count')
ax_arousal.set_xlabel('Speakers')
ax_arousal.legend(title='Arousal')
add_counts_on_bars(ax_arousal, arousal_data)

plt.tight_layout()
plt.show()
