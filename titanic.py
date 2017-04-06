#!/usr/bin/env python

import numpy as np
import tflearn
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv

titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
# data looks like [[some, field, for, a, sample, point...], [some, field, for, another, sample, point,...],...]
# labels looks like [[0, 1], [0, 1], [1, 0],...]
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

def preprocess(data, columns_to_ignore):
    for id in sorted(columns_to_ignore, reverse=True):
        for i in range(len(data)):
            data[i].pop(id)
    for i in range(len(data)):
        # Converting 'sex' field to float (id is 1 after removing labels column)
        data[i][1] = 1 if data[i][1] == 'female' else 0 
    return np.array(data, dtype=np.float32)

columns_to_ignore = [1, 6] # Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
data = preprocess(data, columns_to_ignore)