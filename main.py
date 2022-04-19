# Imports
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from datetime import datetime

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

label_dict = {0: 'air conditioner',
              1: 'car horn',
              2: 'children playing',
              3: 'dog bark',
              4: 'drilling',
              5: 'engine idling',
              6: 'gun shot',
              7: 'jackhammer',
              8: 'siren',
              9: 'street music'
              }

audio_dataset_path = 'UrbanSound8K/audio/'
metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

start = datetime.now()


def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    #mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mel_features = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40, fmax=8000)
    mel_scaled_features = np.mean(mel_features.T, axis=0)
    return mel_scaled_features


extracted_features = []

for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
    final_class_labels = row["class"]
    final_class_folds = row["fold"]
    data = features_extractor(file_name)
    extracted_features.append([data, final_class_labels, final_class_folds])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class', 'fold'])
extracted_features_df.head()

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())
folds = np.array(extracted_features_df['fold'].tolist())

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

num_labels = y.shape[1]

# Model
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', verbose=1, save_best_only=True)

group_kfold = GroupKFold(n_splits=10)
group_kfold.get_n_splits(X, y, folds)

accuracies = []

for train_index, test_index in group_kfold.split(X, y, folds):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(X_train, X_test, y_train, y_test)

    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

    test_accuracy = model.evaluate(X_test,y_test,verbose=0)
    print(test_accuracy[1])
    accuracies.append(test_accuracy[1])

print("accuracies of 10-fold cross validation: " + str(accuracies))

duration = datetime.now() - start
print("Training completed in time: ", duration)


filename = 'UrbanSound8K/audio/fold3/6988-5-0-1.wav'
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

#print(mfccs_scaled_features)
mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)
#print(mfccs_scaled_features)
#print(mfccs_scaled_features.shape)
predicted_label = model.predict(mfccs_scaled_features)
#print(predicted_label)
prediction_class = np.argmax(predicted_label, axis=1)
print("prediction: " + label_dict[prediction_class[0]])
print("actual: " + label_dict[int(filename.split("-")[-3])])