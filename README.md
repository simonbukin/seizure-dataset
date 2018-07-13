# Epileptic Seizure Dataset Analysis

This project attempts to predict the class of seizure a sample has given 178 timestamped EEG readings.

This is done using the Keras library. The structure is relatively simple, consisting of Dense layers with LeakyReLU activations. The final layer is the output classes using softmax activation to predict the class of seizure given the input. After KFold evaluation, the model achieved 76% accuracy on test data.

Jupyter Notebook coming soon!

# API/Dependencies (also available in the Pipfile):

- Keras
- pandas
- sklearn
- tensorflow

This project was run in python 3.5.
