import flask
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
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import jsonify

warnings.filterwarnings("ignore")


def get_MFCC(file_path: str, duration=2):
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

        mfcc_feature = preprocessing.scale(mfcc_feature)
        return mfcc_feature
    except Exception as e:
        print(f"Something went wrong with file {file_path}: cause: {e}")


def predict_gender(model, file_path: str):
    mfcc_feature = get_MFCC(file_path)
    flat = mfcc_feature.flatten().tolist()
    X_real = [flat]
    y_new = model.predict(X_real)
    return y_new[0]


MODEL_PATH = os.path.dirname(__file__) + "\\2021-08-01-185615-model.bin"
MEN = 0
WOMAN = 1   

UPLOAD_FOLDER = 'c:/uploads'
ALLOWED_EXTENSIONS = {'wav'}


filehandler = open(MODEL_PATH, 'rb')
loaded_model = pickle.load(filehandler)

# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            wynik = predict_gender(loaded_model, UPLOAD_FOLDER + "/" + filename)
            os.remove(UPLOAD_FOLDER + "/" + filename)

            if wynik == MEN:
                return jsonify(gender="MALE")
            else:
                return jsonify(gender="FEMALE")


    return '''
    <!doctype html>
    <title>Upload audio wav</title>
    <h1>Upload new audio</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

 

app.run(host='0.0.0.0', port=8000, debug=True)
