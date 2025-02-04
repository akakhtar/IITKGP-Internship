from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

df_train = pd.read_csv('/content/drive/My Drive/CMED/ross/rawTrainTestDev/Ross_train.csv')
df_test = pd.read_csv('/content/drive/My Drive/CMED/ross/rawTrainTestDev/Ross_test.csv')
df_dev = pd.read_csv('/content/drive/My Drive/CMED/ross/rawTrainTestDev/Ross_dev.csv')

df = pd.concat([df_train,df_test, df_dev], ignore_index=True)
print(f"Shape of Combined Data Set : {df.shape}")

df["landmarks"].value_counts().get("{}", 0)
df = df[df["landmarks"] != '{}']
df.shape

train, test = train_test_split(df, test_size=0.2, random_state=42)
print(f"Shape of Train Data Set : {train.shape}")
print(f"Shape of Test Data Set : {test.shape}")

def extract_coordinates(landmarks_str):
  landmarks = ast.literal_eval(landmarks_str)
  coordinates = []
  for landmark in landmarks.values():
    coordinates.append(landmark['X'])
    coordinates.append(landmark['Y'])
  return np.array(coordinates).flatten()

train['landmarks'] = train['landmarks'].apply(extract_coordinates)
test['landmarks'] = test['landmarks'].apply(extract_coordinates)

landmarks_train = []
for index, row in train.iterrows():
    landmarks_train.append(row['landmarks'])

scaler = StandardScaler()
landmarks_scaled_train = scaler.fit_transform(landmarks_train)
print(landmarks_scaled_train)
kpca = KernelPCA(n_components=1, kernel="rbf",eigen_solver = 'arpack')
landmarks_reduced_train = kpca.fit_transform(landmarks_scaled_train)
train["facial_feature"] = landmarks_reduced_train
print(train.shape)

landmarks_test = []
for index, row in test.iterrows():
    landmarks_test.append(row['landmarks'])

landmarks_scaled_test = scaler.transform(landmarks_test)
landmarks_reduced_test = kpca.transform(landmarks_scaled_test)
test["facial_feature"] = landmarks_reduced_test
print(test.shape)

train.to_csv('/content/drive/My Drive/CMED/ross/ross_train.csv', index=False)
test.to_csv('/content/drive/My Drive/CMED/ross/ross_test.csv', index=False)
    
