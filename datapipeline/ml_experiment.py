import numpy as np
import pandas as pd
import os
import json, gc
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import KFold
import pickle
import matplotlib.pyplot as plt

path_to_train = ''
path_to_save = ''

def get_x_y_floor(truth_array):
    xs = list()
    ys = list()
    floors = list()
    for lst in truth_array:
        xs.append(lst[0])
        ys.append(lst[1])
        floors.append(lst[2])
    return xs,ys,floors

def gen_for_serialisation(path_to_train):
    site_data_files = [x for x in os.listdir(path_to_train)]
    for file in site_data_files:
        #../input/indoor-positioning-traindata/train_data
        f = open(path_to_train +'/'+file, "rb")
        site, train, ground = pickle.load(f)
        f.close()
        yield site,train,ground

def create_model2(input_d):
    w1 = int(input_d/2)
    w2 = int(input_d/4)
    model = keras.Sequential()
    model.add(layers.Dense(w1, activation='relu'))
    model.add(layers.Dense(w2, activation='sigmoid'))
    model.add(layers.Dense(w2, activation='sigmoid'))
    model.add(layers.Dense(3))
    model.compile(optimizer=tf.optimizers.Adam(lr=0.001),
                  loss='mse', metrics=['mse'])
    #model.summary()
    return model

    #srun --gres=gpu:1 --pty singularity shell --nv tensorflow_19.03-py3.sif

def fit_model_site(site, train_data, target_data, path_to_save):
    #xs,ys,floors = get_x_y_floor(target_data)
    train_data = np.array(train_data)
    train_data = train_data.astype(np.float)
    target_data = np.array(target_data)
    target_data = target_data.astype(np.float)
    #predictors = ['x','y','floor']
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    
    for train, test in kfold.split(train_data, target_data):
        model = create_model2(train_data[0].shape[0])
        history = model.fit(train_data[train], target_data[train], verbose=1, epochs=50, validation_data=(train_data[test],target_data[test]))
        with open('{pts}/{site}_{fold}_NN02.pickle'.format(pts=path_to_save,site=site, fold=fold_no), 'wb') as f:
            pickle.dump(history.history, f)

if __name__ == "__main__":
    gen = gen_for_serialisation()
    for site, train, truth in gen:
        fit_model_site(site,train, truth)