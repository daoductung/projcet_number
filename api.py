import librosa
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
from handling_file import *
import pandas as pd
import pickle
from annoy import AnnoyIndex
from collections import Counter

data_dir = './data'


def extract_features(y, sr=16000, nfilt=10, winsteps=0.03):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        raise Exception("Extraction feature error")


def crop_feature(feat, i=0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i: i + nb_step]).flatten()
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    return crop_feat


path_file_title = path_root + '\\title.csv'


def read_file_title(path_file_title):
    meta_data = pd.read_csv(path_file_title)
    return meta_data


# Todo: Read pandas

def read_data_file_wav(path_file_title):
    features = list()
    number = list()
    metadata = read_file_title(path_file_title)
    for index, row in metadata.iterrows():
        y, sr = librosa.load(path_train + row['file_name'], sr=16000)
        feat = extract_features(y)
        for i in range(0, feat.shape[0] - 10, 5):
            features.append(crop_feature(feat, i, nb_step=10))
            number.append(row['classID'])
    return features, number


features, number = read_data_file_wav(path_file_title)

pickle.dump(features, open('pickle/features.pk', 'wb'))

pickle.dump(number, open('pickle/songs.pk', 'wb'))


def train(features):
    f = 100
    t = AnnoyIndex(f, metric='angular')

    for i in range(len(features)):
        v = features[i]
        t.add_item(i, v)
    t.build(100)
    t.save('ann/number.ann')


def load_ann(f=100):
    u = AnnoyIndex(f, metric='angular')

    u.load('ann/number.ann')
    return u


def detect(file_record, u):
    path = path_record + file_record
    y, sr = librosa.load(path, sr=16000)
    feat = extract_features(y)
    results = []
    for i in range(0, feat.shape[0], 10):
        crop_feat = crop_feature(feat, i, nb_step=10)
        result = u.get_nns_by_vector(crop_feat, n=5)
        result_songs = [number[k] for k in result]
        results.append(result_songs)
    results = np.array(results).flatten()
    most_song = Counter(results)
    print(most_song.most_common())
    return most_song.most_common()[0][0]

# train(features)
