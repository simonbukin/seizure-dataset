import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show

dataset = pd.read_csv('seizure_data.csv') # load in seizure dataset

dataset = dataset.drop(columns=['Unnamed: 0'], axis=1) # clean columns

X = dataset.drop(columns=['y'])
Y = dataset['y']

output_file('multiline.html')
plot = figure(title="EEG Values over Time", x_axis_label='Time', y_axis_label='EEG Value')

sum_series = pd.Series() # series summer
sum_seizures = pd.Series()
sum_non_seizures = pd.Series()
seizures = 0
for index, row in X.iterrows(): # iterate through all rows
    if Y[index] == 1: # sum seizures
        sum_seizures = sum_seizures.add(row, fill_value=0)
        seizures += 1
    else: # sum non-seizures
        sum_non_seizures = sum_non_seizures.add(row, fill_value=0)
    sum_series = sum_series.add(row, fill_value=0)

sum_series = sum_series.divide(X.shape[0]) # find average values
sum_seizures = sum_seizures.divide(seizures)
sum_non_seizures = sum_non_seizures.divide(X.shape[0] - seizures)

plot.line(np.linspace(0, 1, 178), sum_series, line_color='green')
plot.line(np.linspace(0, 1, 178), sum_seizures, line_color='red')
plot.line(np.linspace(0, 1, 178), sum_non_seizures, line_color='blue')

show(plot)
