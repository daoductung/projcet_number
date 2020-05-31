from handling_file import *
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from helper.wavfilehelper import WavFileHelper
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from datetime import datetime

# todo: Hien thi bieu do am thanh cua file
file_name = path_train + '0\\00.wav'


def show_wav(file_name):
    file_name = path_train + '0\\00.wav'
    plt.figure(figsize=(12, 4))
    data, sample_rate = librosa.load(file_name)
    _ = librosa.display.waveplot(data, sr=sample_rate)
    ipd.Audio(file_name)
    plt.show()


# Todo: Read pandas
path_file_title = path_root + '\\title.csv'


def read_file_title(path_file_title):
    meta_data = pd.read_csv(path_file_title)
    return meta_data


# Todo: Count class
def count_class(meta_data):
    return meta_data.className.value_counts()


# print(count_class(read_file_title(path_file_title)))

def audio_poperties():
    wavfilehelper = WavFileHelper()
    metadata = read_file_title(path_file_title)
    audiodata = []
    for index, row in metadata.iterrows():
        data = wavfilehelper.read_file_properties(path_train + row['file_name'])
        audiodata.append(data)

    audiodf = pd.DataFrame(audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])
    return audiodf


audiodf = audio_poperties()


# print(audio_poperties())
# print(audiodf.num_channels.value_counts(normalize=True))
# print(audiodf.sample_rate.value_counts(normalize=True))
# bit depth
# print(audiodf.bit_depth.value_counts(normalize=True))

# librosa_audio, librosa_sample_rate = librosa.load(file_name)
# scipy_sample_rate, scipy_audio = wav.read(file_name)
#
# mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
# print(mfccs.shape)
# librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled


# Load various imports

def load_various_import(file_title):
    features = []

    # Iterate through each sound file and extract the features
    for index, row in read_file_title(file_title).iterrows():
        class_label = row["className"]
        data = extract_features(path_train + row['file_name'])

        features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    # print('Finished feature extraction from ', len(featuresdf), ' files')
    return featuresdf


featuresdf = load_various_import(path_file_title)
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

num_labels = yy.shape[1]
filter_size = 2

# Construct model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# Display model architecture summary
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100 * score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath=path_root + '\\save_models\\weights.hd5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
          callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)

score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


def extract_feature(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    return np.array([mfccsscaled])


def print_prediction(file_name):
    prediction_feature = extract_feature(file_name)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))


filename = path_root + '\\test\\test_1\\0.wav'
print_prediction(filename)
