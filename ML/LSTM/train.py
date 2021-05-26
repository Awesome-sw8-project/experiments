import numpy as np, pickle
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import KFold

from ML.LightGBM import comp_metric
from ML.LSTM.models.model import create_model

#target data for time series
def get_target_data(ground):
    xs,ys,floors = list(),list(),list()
    for x in ground:
        subx,suby,subf =list(),list(),list()
        for i in x:
            subx.append([i[0]])
            suby.append([i[1]])
            subf.append([i[2]])
        xs.append(np.asarray(subx))
        ys.append(np.asarray(suby))
        floors.append(np.asarray(subf))
        subx.clear()
        suby.clear()
        subf.clear()
    return xs,ys,floors

def fit(site, train_data, target_data, experiment_no, path_to_save, test_data,mx_step):
    train = np.array(train_data, dtype="object")
    train = [np.array(x).astype(np.float32) for x in train]
    ground = np.array(target_data, dtype="object")
    ground = [np.array(x).astype(np.float32) for x in ground]
    processed_train = tf.keras.preprocessing.sequence.pad_sequences(train, padding="post",maxlen=mx_step, value=0.0)
    processed_ground = tf.keras.preprocessing.sequence.pad_sequences(ground, padding="post",maxlen=mx_step, value=0.0)
    predictions = dict()
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(processed_train, processed_ground):
        model = create_model(processed_train.shape)
        history = model.fit(
                            processed_train[train], 
                            processed_ground[train], 
                            verbose=2, 
                            epochs=100,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
                            validation_data=(processed_train[test],processed_ground[test]))
        with open('{pts}/{site}_{fold}_LSTM{exp_no}.pickle'.format(pts=path_to_save,site=site, fold=fold_no, exp_no=experiment_no), 'wb') as f:
            pickle.dump(history.history, f)
        fold_no +=1
    return model.predict(processed_train[test]).flatten(),processed_ground[test].flatten()
    

def fit_model_site_all_three_model(site, train_data, target_data, path_to_save, exp_no, test_data,mxstep):
    xs,ys,floors = get_target_data(target_data)
    target_data = {"xs": xs, "ys": ys, "floors" :floors}
    test_preds = dict()
    test_ground = dict()
    for target in ["xs", "ys", "floors"]:
        test_preds[target],test_ground[target] = fit(site,train_data, target_data[target], "LSTM{exp_no}_{name}.pickle".format(exp_no=exp_no, name=target), path_to_save, test_data,mxstep)
    mpe = comp_metric(test_preds["xs"], test_preds["ys"], test_preds["floors"], test_ground["xs"], test_ground["ys"], test_ground["floors"])
    print("MPS for site: {} is {}".format(site,mpe))
    with open("mpe.text","a") as f:
        f.write("{}: {}\n".format(site, mpe))