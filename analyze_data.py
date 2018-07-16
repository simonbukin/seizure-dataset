import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import Adagrad
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('seizure_data.csv') # load in seizure dataset

# the X1-X178 data represents explanatory variables corresponding to EEG values of individual patients
# the y value is the class that we are classifying. 1 is epileptic seizure, 2-5 are different states.

dataset = dataset.drop(columns=['Unnamed: 0'], axis=1)

X = dataset.drop(columns=['y'])
Y = dataset['y']

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
categorical_Y = np_utils.to_categorical(encoded_Y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

def baseline_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=178, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adagrad(), metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=64, verbose=1)

kfold = KFold(n_splits=5, shuffle=True, random_state=3)

results = cross_val_score(estimator, X, categorical_Y, cv=kfold)
print('Accuracy: {} ({})'.format(results.mean() * 100)
