**Project Overview**

Dataset Preparation

Dataset:

We utilized the MELD dataset, which is organized into three folders: Train, Dev, and Test.

Splitting:

The dataset was divided based on the main six speakers: Joey, Ross, Rachel, Phoebe, Monica, and Chandler.
Each of the three folders (Train, Dev, and Test) was split into six files, one for each of the main speakers.

Data Processing

Mapping and Sequence Numbering:

We performed mapping of emotions and allocated sequence numbers for dialogues.

Generation of Self-Report Dataset:

Calculated Influence_0, Influence_1, and Sequence Length for each speaker in each folder (Train, Dev, Test).

Feature Extraction

Facial Analysis:

Extraction:

Used OpenCV and AWS Rekognition to extract facial landmarks from video frames.
Feature Reduction: Reduced facial features to a single set of landmarks per utterance.

Audio Analysis:

Extraction: Extracted audio features using librosa, including MFCCs, Mel Spectrogram, and Spectral Contrast.
Feature Reduction: Applied Kernel Principal Component Analysis (KPCA) to reduce audio features to a single representative feature.

Data Integration and Model Building

Data Integration:

Combined the datasets from Train, Dev, and Test folders, maintaining the same speakers.
Split the combined dataset into training and testing sets.

Feature Reduction:

Applied feature reduction techniques to both facial and audio features.

Model Building:

Built a predictive model using Random Forest Classifier.
Evaluated model performance using classification metrics such as precision, recall, and F1-score.

Model Evaluation

Performance Analysis:

Analyzed the model's performance through classification reports and confusion matrices.
Conducted cross-validation with 5 folds to assess the model’s stability and generalizability.
