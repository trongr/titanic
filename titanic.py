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

