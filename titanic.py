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

net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# Testing
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

dicaprio, winslet = preprocess([dicaprio, winslet], columns_to_ignore)

pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])