import os
import librosa
import pickle
import numpy as np
from sklearn import mixture
from sklearn import preprocessing
from datetime import datetime
import warnings
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys

warnings.filterwarnings("ignore")


def get_MFCC(file_path: str, duration = 2):
    try:
        n_mfcc = 13
        n_mels = 40
        n_fft = 512 
        hop_length = 160
        fmin = 0
        fmax = None
        sr = 16000

        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        mfcc_feature = librosa.feature.mfcc(y=audio, sr=sr, n_fft=n_fft,
                                    n_mfcc=n_mfcc, n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin, fmax=fmax, htk=False)

        mfcc_feature  = preprocessing.scale(mfcc_feature)
        return mfcc_feature
    except Exception as e:
        print(f"Something went wrong with file {file_path}: cause: {e}" )

def predict_gender(model, file_path: str):
    mfcc_feature = get_MFCC(file_path)
    flat = mfcc_feature.flatten().tolist()
    X_real = [flat]
    y_new = model.predict(X_real)
    return y_new[0]  


MODEL_PATH =  os.path.dirname(__file__) + "\\2021-08-01-185615-model.bin"
MEN = 0
WOMAN = 1


if __name__ == "__main__":
    file_to_test = sys.argv[1]
    print(f"predict gender for file: {file_to_test}")

    filehandler = open(MODEL_PATH, 'rb')
    loaded_model = pickle.load(filehandler)  

    wynik = predict_gender(loaded_model, file_to_test)
    if wynik == MEN:
        print("mezczyzna")
    else:
        print("kobieta")      