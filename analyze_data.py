import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
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
dummy_Y = np_utils.to_categorical(encoded_Y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# X_train = X.sample(frac=0.2,random_state=200)
# X = X.drop(X_train.index)
#
# Y_train = Y.sample(frac=0.2,random_state=200)
# Y = Y.drop(Y_train.index)

def baseline_model():
    model = Sequential()
    model.add(Dense(100, input_dim=178))
    for i in range(0, 10):
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        model.add(Dense(150))

    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=150, batch_size=150, verbose=1)

kfold = KFold(n_splits=5, shuffle=True, random_state=3)

results = cross_val_score(estimator, X, dummy_Y, cv=kfold)
print('Baseline: {} ({})'.format(results.mean()*100, results.std()*100))
