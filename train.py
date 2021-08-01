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

warnings.filterwarnings("ignore")
#https://github.com/SuperKogito/Voice-based-gender-recognition  ??

MEN = 0
WOMAN = 1

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


def load_gender_feature(genderY: int, source_path: str):
    files = [os.path.join(source_path, f)
         for f in os.listdir(source_path) if f.endswith('.wav')]
    results = []

    for f in files:
        mfcc_feature = get_MFCC(f)
        flat = mfcc_feature.flatten().tolist()
        flat.append(genderY)
        results.append(flat)
    return results


def combine_genders(men, woman, shuffle_rows = False ):
    columns_names = []
    for i in range(0, len(men[0])-1):
        columns_names.append(f'x{i}')
    columns_names.append('y')

    men_df = pd.DataFrame(men, columns = columns_names)
    woman_df = pd.DataFrame(woman, columns = columns_names)
    combined = pd.concat([men_df, woman_df], ignore_index=True, sort=False)
    
    if shuffle_rows:
        combined = shuffle(combined)
 
    combined = combined.reset_index()
    return combined

def predict_gender(model, file_path: str):
    mfcc_feature = get_MFCC(file_path)
    flat = mfcc_feature.flatten().tolist()
    X_real = [flat]
    y_new = model.predict(X_real)
    return y_new[0]    

#=================================================================================================
print("LOAD FILES")        

# path to training data
dirname = os.path.dirname(__file__)
male_source = os.path.join(dirname, "data\\train_data\\AudioSet\\male_clips\\")
female_source = os.path.join(dirname, "data\\train_data\\AudioSet\\female_clips\\")

male_data = load_gender_feature(MEN, male_source)
female_data = load_gender_feature(WOMAN, female_source)

gender_df = combine_genders(male_data, female_data, True)
 
#=================================================================================================
print("CLEAN DATA")

gender_df = gender_df.dropna()
gender_df.drop('index', axis=1, inplace=True)
#print(gender_df.head(n=10))

#=================================================================================================
print("TRAIN")     

feature_names = [col for col in gender_df.columns]
feature_names = feature_names[:-1]

train, test = train_test_split(gender_df, test_size = 0.4, random_state = 50)

X_train = train[feature_names]
y_train = train['y']

X_test = test[feature_names]
y_test = test['y']

mod_dt = DecisionTreeClassifier(max_depth = 5, random_state = 50)
mod_dt.fit(X_train,y_train)
prediction = mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction, y_test)))

mod_svc = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf', random_state = 50))
mod_svc.fit(X_train, y_train)
prediction = mod_svc.predict(X_test)
print('The accuracy of the SVC is',"{:.3f}".format(metrics.accuracy_score(prediction, y_test)))


model_to_save = mod_svc

#=================================================================================================
print("SAVE MODEL")
prefix_day = datetime.now().strftime("%Y-%m-%d")   
prefix_time = datetime.now().strftime("%H%M%S")

picklefile = f"\\{prefix_day}-{prefix_time}-model.bin"
filehandler = open(dirname+picklefile, 'wb')
pickle.dump(model_to_save, filehandler)


#=================================================================================================
print("LOAD MODEL") 
filehandler = open(dirname+picklefile, 'rb')
loaded_model = pickle.load(filehandler)  


#=================================================================================================
print("REAL PREDICTION TEST") 

dirname = os.path.dirname(__file__)
test_file = dirname + '\\test_data\\woman.wav'

wynik = predict_gender(loaded_model, test_file)
if wynik == MEN:
    print("mezczyzna")
else:
    print("kobieta")    
   
#=================================================================================================
print("DONE")      
