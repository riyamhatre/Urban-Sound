# Imports
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from datetime import datetime

import tensorflow as tf
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

model = tf.keras.models.load_model('saved_models/audio_classification.hdf5')

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
"""
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(model.predict(test_images).shape)
"""


x = input('Enter your name:')
print('Hello, ' + str(x))

