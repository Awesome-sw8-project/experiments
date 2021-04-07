import os, json, gc
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from load_data import get_x_y_floor, gen_for_serialisation

#change this according to model
from models.model02 import create_model

#fit to the neural net
def fit(train_data, target_data, experiment_no, path_to_save, test_data):
    predictions = dict()
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(train_data, target_data):
        model = create_model(train_data[0].shape[0])
        history = model.fit(
                            train_data[train], 
                            target_data[train], 
                            batch_size=32,
                            verbose=1, 
                            epochs=100,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                            validation_data=(train_data[test],target_data[test]))
        with open('{pts}/{site}_{fold}_NN{exp_no}.pickle'.format(pts=path_to_save,site=site, fold=fold_no, exp_no=experiment_no), 'wb') as f:
            pickle.dump(history.history, f)
        if fold_no == 1:
            for feat, timestamp in test_data:
                predictions[timestamp] = [model.predict(feat)]
        else:
            for feat, timestamp in test_data:
                predictions[timestamp] = np.add(predictions[timestamp], model.predict(feat))

        fold_no = fold_no +1
    for key in predictions.keys():
        predictions[key] = [x/10 for x in predictions[key]]
    return predictions

#fits model to x,y and floor coordinates
def fit_model_site(site, train_data, target_data, path_to_save, test_data):
    train_data = np.array(train_data)
    train_data = train_data.astype(np.float)
    target_data = np.array(target_data)
    target_data = target_data.astype(np.float)
    #predictors = ['x','y','floor']
    fit(train_data, target_data, "03", path_to_save, test_data)

#fits models for x,y, and floor seperately.
def fit_model_site_all_three_model(site, train_data, target_data, path_to_save, exp_no, test_data)
    train_data = np.array(train_data)
    xs,ys,floors = get_x_y_floor(target_data)
    target_data = {"xs": xs, "ys": ys, "floors" :floors}
    for target in ["xs", "ys", "floors"]:
        fit(train_data, target_data[target], "NN{exp_no}_{name}.pickle".format(exp_no=exp_no, name=target), path_to_save, test_data)


if __name__ == "__main__":
    gen = gen_for_serialisation()
    for site, train, truth in gen:
        fit_model_site(site,train, truth)