Steps: 
1)** Mapping**: emotions to valence and arousal, also ignoring where emotion is 'neutral' and 'surprise'.

2)** Splitting**: splitting of the CSV file based on the 6 speakers for each of the three folders.

3) **Facial Feature Extraction**: for each speaker, their facial features are captured and stored in the column 'landmarks'.
   
4) **Audio Feature Extraction**: similar to facial feature extraction, we extract 3 audio features such as 'MFCCs', 'mel spectrogram', and 'spectral contrast'.
   
5) **Concatenating of CSV file**: the three CSV files for each speaker named as train, dev, and test are concatenated and sorted based on the dialogue ID and utterance ID. We also assign the scene ID for the same dialogue ID.

6) **Influence and Sequence Length**: we find the self-report features such as influence0, influence1, and sequence length.

7) **Facial Feature Reduction**: first, the joint dataset is broken into train and test without any randomization to maintain the sequence. Then, using KPCA, the facial feature is reduced to a single feature.

8) **Audio Feature Reduction**: similar to facial feature reduction, we reduce the audio to a single feature.

9) **Model Building and Evaluation**: a model is built using a random forest classifier and evaluated for different features.

**Order of Code to run** : mapping.py -> splitting_speakers.py -> facial_feature_extraction.py -> audio_feature_extraction.py -> test_concate.py -> influence_valence.py -> facial_feature_reduction.py -> audio_feature_reduction.py -> building_testing.py

Final csv file for speaker ross: 'ross_test.csv' and 'ross_train.csv' in the folder 'Self Report Features'
