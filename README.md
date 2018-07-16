# Epileptic Seizure Dataset Analysis

This project attempts to predict the class of seizure a sample has given 178 timestamped EEG readings.

This is done using the Keras library. The structure is relatively simple, consisting of Dense layers with Dropouts in between. The final layer is the output classes using sigmoid activation to predict the class of seizure given the input. After KFold evaluation, the model achieved 89% accuracy on test data.

Jupyter Notebook coming soon!

# API/Dependencies (also available in the Pipfile):

- Keras
- pandas
- sklearn
- tensorflow
- jupyter

This project was run in python 3.5.

The dataset used can be found here: http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
