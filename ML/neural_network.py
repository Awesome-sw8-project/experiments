import os
import sys
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import LeaveOneOut,KFold

# import datapipeline to retrieve formatted data
sys.path.insert(0, "../datapipeline/")
from datapipeline import *
# TODO: Add right functionality to retrieve data from data pipeline

# we use data pipeline to retrieve RSSID values
train = [[-999, 34, 36], [16, -999, 43], [26, -999, 43], [36, -999, 43], [46, -999, 43], [56, -999, 43], [66, -999, 43], [76, -999, 43], [86, -999, 43], [96, -999, 43], [106, -999, 43]]
# we use data pipeline to then retrieve the ground truth data corresponding to the train array
# format: [x,y, floor]
ground_truth = [[1, 2, 3], [11, 4, 7], [2, 4, 7], [3, 4, 7], [4, 4, 7], [5, 4, 7], [6, 4, 7], [7, 4, 7], [8, 4, 7], [9, 4, 7], [10, 4, 7]]


def make_dataset(X_data,y_data,n_splits):
    def gen():
        for train_index, test_index in KFold(n_splits).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train,y_train,X_test,y_test

    return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64))

dataset=make_dataset(X,y,10)