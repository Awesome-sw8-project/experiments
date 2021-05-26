import os, json, gc, pickle
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0,"../")
from load_data import get_x_y_floor, gen_for_serialisation, get_data_for_test

#change this according to model
from models.model02 import create_model

pt_testset= ''
path_to_save =''
path_to_train = ""
path_to_sample = ""
#fit to the neural net
def fit(train_data, target_data, experiment_no, path_to_save, test_data):
    target_data = np.asarray(target_data).astype(np.float)
    train_data = np.asarray(train_data).astype(np.float)
    predictions = dict()
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(train_data, target_data):
        model = create_model(train_data[0].shape)
        history = model.fit(
                            train_data[train], 
                            target_data[train], 
                            batch_size=32,
                            verbose=1, 
                            epochs=100,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
                            validation_data=(train_data[test],target_data[test]))
        with open('{pts}/{site}_{fold}_NN{exp_no}.pickle'.format(pts=path_to_save,site=site, fold=fold_no, exp_no=experiment_no), 'wb') as f:
            pickle.dump(history.history, f)
        if fold_no == 1:
            for feat, timestamp in test_data:
                feat1 = np.asarray(feat).astype(np.float)
                predictions[timestamp] = model.predict(np.array([feat1,]))[0]
        else:
            for feat, timestamp in test_data:
                feat1 = np.asarray(feat).astype(np.float)
                predictions[timestamp] = np.add(predictions[timestamp], model.predict(np.array([feat1,]))[0] )

        fold_no = fold_no +1
    for key in predictions.keys():
        predictions[key] = [x/10 for x in predictions[key]]
    return predictions

#Deprecated for Now!!!
#fits model to x,y and floor coordinates
def fit_model_site(site, train_data, target_data, path_to_save, test_data):
    train_data = np.array(train_data)
    train_data = train_data.astype(np.float)
    target_data = np.array(target_data)
    target_data = target_data.astype(np.float)
    fit(train_data, target_data, "03", path_to_save, test_data)

#fits models for x,y, and floor seperately.
def fit_model_site_all_three_model(site, train_data, target_data, path_to_save, exp_no, test_data):
    train_data = np.array(train_data)
    xs,ys,floors = get_x_y_floor(target_data)
    target_data = {"xs": xs, "ys": ys, "floors" :floors}
    test_df = None
    for target in ["xs", "ys", "floors"]:
        preds = fit(train_data, target_data[target], "NN{exp_no}_{name}.pickle".format(exp_no=exp_no, name=target), path_to_save, test_data)
        if target == "xs":
            test_df = pd.DataFrame.from_dict(preds, orient='index', columns=['x'])
            test_df["timestamp"] = test_df.index
        elif target == "ys":
            test_df["y"] = test_df["timestamp"].map(preds)
            test_df["y"] = test_df["y"].apply(lambda x: x[0])
        elif target == "floors":
            test_df["floor"] = test_df["timestamp"].map(preds)
            test_df["floor"] = test_df["floor"].apply(lambda x: x[0])
    return test_df

def get_sample_submission_index(path_to_sample):
    df = pd.read_csv(path_to_sample)
    return df["site_path_timestamp"]

def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat-x, 2) + np.power(yhat-y, 2)) + 15 * np.abs(fhat-f)
    return intermediate.sum() / xhat.shape[0]

#fits models for x,y, and floor seperately.
def fit_models_site(site, train_data, target_data,val_feat,val_ground, path_to_save, exp_no, test_data):
    train_data = np.asarray(train_data).astype(np.float)
    xs,ys,floors = get_x_y_floor(target_data)
    target_data = np.asarray(target_data).astype(np.float)
    model = create_model(train_data.shape)
    history = model.fit(
                        train_data, 
                        xs, 
                        batch_size=32,
                        verbose=1, 
                        epochs=100,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
                        metrics=['root_mean_squared_error'],
                        validation_data=(val_feat,val_ground))
    with open("{}/X/".format(path_to_save,),"wb") as f:
        pickle.dump(history,f)
    x_preds = model.predict(train_data)
    model = create_model(train_data.shape)
    history = model.fit(
                        train_data, 
                        ys, 
                        batch_size=32,
                        verbose=1, 
                        epochs=100,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
                        metrics=['root_mean_squared_error'],
                        validation_data=(val_feat,val_ground))
    with open("{}/Y/".format(path_to_save,),"wb") as f:
        pickle.dump(history,f)
    y_preds = model.predict(train_data)
    model = create_model(train_data.shape)
    history = model.fit(
                        train_data, 
                        floors, 
                        batch_size=32,
                        verbose=1, 
                        epochs=100,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
                        metrics=['root_mean_squared_error'],
                        validation_data=(val_feat,val_ground))
    with open("{}/X/".format(path_to_save,),"wb") as f:
        pickle.dump(history,f)
    floor_preds = model.predict(train_data)
    msp = comp_metric(x_preds,y_preds,floor_preds,xs,ys,floors)
    print(msp)


if __name__ == "__main__":
    sample_dfs = list()
    gen = gen_for_serialisation(path_to_train)
    for site, train, truth in gen:
        test_data = get_data_for_test(pt_testset, site)
        sample_dfs.append( fit_model_site_all_three_model(site,train, truth, path_to_save, "04", test_data))
        #fit_model_site_all_three_model(site,train, truth, path_to_save, "04", test_data)
        break
    sample_df = pd.concat(sample_dfs)
    index = get_sample_submission_index(path_to_sample)
    sample_df = sample_df.reindex(index).fillna(0)
    sample_df.to_csv("res.csv")