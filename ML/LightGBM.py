# Need a input folder in the root folder with the data without min max
# Need a folder called LightGBM_models for the models
# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import glob

from sklearn.model_selection import KFold
import lightgbm as lgb

import psutil
import random
import os
import time
import sys
import math
from contextlib import contextmanager
import joblib

from load_data import *


# ------------------------------------------------------------------------------
# Fixed values
# ------------------------------------------------------------------------------
N_SPLITS = 5
SEED = 42

# ------------------------------------------------------------------------------
# File path definition
# ------------------------------------------------------------------------------
LOG_PATH = Path("./log/")
LOG_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
@contextmanager
def timer(name: str):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    try:
        yield
    finally:
        m1 = p.memory_info()[0] / 2. ** 30
        delta = m1 - m0
        sign = '+' if delta >= 0 else '-'
        delta = math.fabs(delta)
        print(f"[{m1:.1f}GB({sign}{delta:.1f}GB): {time.time() - t0:.3f}sec] {name}", file=sys.stderr)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    
def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat-x, 2) + np.power(yhat-y, 2)) + 15 * np.abs(fhat-f)
    return intermediate.sum() / xhat.shape[0]


def score_log(df: pd.DataFrame, num_files: int, nam_file: str, data_shape: tuple, n_fold: int, seed: int, mpe: float):
    score_dict = {'n_files': num_files, 'file_name': nam_file, 'shape': data_shape, 'fold': n_fold, 'seed': seed, 'score': mpe}
    df = pd.concat([df, pd.DataFrame.from_dict([score_dict])])
    df.to_csv(LOG_PATH / f"log_score.csv", index=False)
    return df

def create_submission_file(predictions):
    all_preds = pd.concat(predictions)
    all_preds = all_preds.reindex(subm.index)
    all_preds.to_csv('submission.csv')

# ------------------------------------------------------------------------------
# Set seed
# ------------------------------------------------------------------------------
set_seed(SEED)

# ------------------------------------------------------------------------------
# Input data
# ------------------------------------------------------------------------------
train_files = sorted(glob.glob('../input/indoor-positioning-data/train_data/*'))
test_files = sorted(glob.glob('../input/indoor-positioning-data/test_data/*'))
subm = pd.read_csv('../input/indoor-positioning-data/sample_submission.csv', index_col=0)

# ------------------------------------------------------------------------------
# Define parameters for models
# ------------------------------------------------------------------------------
lgb_params = {'objective': 'root_mean_squared_error',
              'boosting_type': 'gbdt',
              'n_estimators': 50000,
              'learning_rate': 0.1,
              'num_leaves': 90,
              'colsample_bytree': 0.4,
              'subsample': 0.7,
              'subsample_freq': 2,
              'bagging_seed': SEED,
              'reg_alpha': 8,
              'reg_lambda': 2,
              'random_state': SEED,
              'n_jobs': -1
              }

lgb_f_params = {'objective': 'multiclass',
                'boosting_type': 'gbdt',
                'n_estimators': 50000,
                'learning_rate': 0.1,
                'num_leaves': 90,
                'colsample_bytree': 0.4,
                'subsample': 0.6,
                'subsample_freq': 2,
                'bagging_seed': SEED,
                'reg_alpha': 10,
                'reg_lambda': 2,
                'random_state': SEED,
                'n_jobs': -1
                }


# ------------------------------------------------------------------------------
# Create the models
# ------------------------------------------------------------------------------
def create_models():
    score_df = pd.DataFrame()
    oof = list()

    for n_files, file in enumerate(train_files):

        # Check if the model files exist, in case all exist, then skip site
        new_x = False
        new_y = False
        new_f = False

        if not(os.path.exists(f'LightGBM_models/{n_files}modelx.txt')):
                new_x = True
        if not(os.path.exists(f'LightGBM_models/{n_files}modely.txt')):
                new_y = True
        if not(os.path.exists(f'LightGBM_models/{n_files}modelf.txt')):
                new_f = True

        if not(new_x) and not(new_y) and not(new_f):
            print(f"Skipping site {n_files}")
            continue

        # init variables
        site, train, ground = gen_for_serialisation_one_path(file)
        df1 = pd.DataFrame(train)
        df2 = pd.DataFrame(ground)
        data = pd.concat([df1, df2], axis=1, join='inner')

        oof_x, oof_y, oof_f = np.zeros(data.shape[0]), np.zeros(data.shape[0]), np.zeros(data.shape[0])

        # Iterate through each k-fold
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        for fold, (trn_idx, val_idx) in enumerate(kf.split(data.iloc[:, :-3])):
            # Structure the trainings data and validation data
            X_train = data.iloc[trn_idx, :-3].astype(int)
            y_trainx = data.iloc[trn_idx, -3]
            y_trainy = data.iloc[trn_idx, -2]
            y_trainf = data.iloc[trn_idx, -1]

            X_valid = data.iloc[val_idx, :-3].astype(int)
            y_validx = data.iloc[val_idx, -3]
            y_validy = data.iloc[val_idx, -2]
            y_validf = data.iloc[val_idx, -1]
            
            # Check for each model and creates a new if one is missing
            if new_x:
                new_x = True
                modelx = lgb.LGBMRegressor(**lgb_params)
                with timer("fit X"):
                    modelx.fit(X_train, y_trainx,
                               eval_set=[(X_valid, y_validx)],
                               eval_metric='rmse',
                               verbose=False,
                               early_stopping_rounds=20
                               )
                modelx.booster_.save_model(f'LightGBM_models/{n_files}modelx.txt')
            else:
                modelx = lgb.Booster(model_file=f'LightGBM_models/{n_files}modelx.txt')

            if new_y:
                new_y = True
                modely = lgb.LGBMRegressor(**lgb_params)
                with timer("fit Y"):
                    modely.fit(X_train, y_trainy,
                               eval_set=[(X_valid, y_validy)],
                               eval_metric='rmse',
                               verbose=False,
                               early_stopping_rounds=20
                               )
                modely.booster_.save_model(f'LightGBM_models/{n_files}modely.txt')
            else:
                modely = lgb.Booster(model_file=f'LightGBM_models/{n_files}modely.txt')

            if new_f:
                new_f = True
                modelf = lgb.LGBMClassifier(**lgb_f_params)
                with timer("fit F"):
                    modelf.fit(X_train, y_trainf,
                               eval_set=[(X_valid, y_validf)],
                               eval_metric='multi_logloss',
                               verbose=False,
                               early_stopping_rounds=20
                               )
                joblib.dump(modelf, f'LightGBM_models/{n_files}modelf.txt')
            else:
                modelf = joblib.load(f'LightGBM_models/{n_files}modelf.txt')
                
            # Make predictions on validation data and compare to ground truth and add to log file
            oof_x[val_idx] = modelx.predict(X_valid)
            oof_y[val_idx] = modely.predict(X_valid)
            oof_f[val_idx] = modelf.predict(X_valid).astype(int)

            score = comp_metric(oof_x[val_idx], oof_y[val_idx], oof_f[val_idx],
                                y_validx.to_numpy(), y_validy.to_numpy(), y_validf.to_numpy())
            print(f"fold {fold}: mean position error {score}")
            score_df = score_log(score_df, n_files, os.path.basename(file), data.shape, fold, SEED, score)

        # Calculate the MPE for the enitre site and add info to log file
        print("*+"*40)
        print(f"file #{n_files}, shape={data.shape}, name={os.path.basename(file)}")
        score = comp_metric(oof_x, oof_y, oof_f,
                            data.iloc[:, -3].to_numpy(), data.iloc[:, -2].to_numpy(), data.iloc[:, -1].to_numpy())
        oof.append(score)
        print(f"mean position error {score}")
        print("*+"*40)
        score_df = score_log(score_df, n_files, os.path.basename(file), data.shape, 999, SEED, score)

def predict():

    predictions = list()

    # Iterate through each site
    for n_files, file in enumerate(test_files):

        # init variables
        data = gen_for_serialisation_one_path_test(file)
        data = pd.DataFrame(data)
        data.columns = ['Bssid','site_path_timestamp']
        data = pd.concat([pd.DataFrame(i for i in data['Bssid']).astype(int),data['site_path_timestamp']], axis=1)

        preds_x, preds_y, preds_f = 0, 0, 0

        #Load models for relavant site
        modelx = lgb.Booster(model_file=f'LightGBM_models/{n_files}modelx.txt')
        modely = lgb.Booster(model_file=f'LightGBM_models/{n_files}modely.txt')
        modelf = joblib.load(f'LightGBM_models/{n_files}modelf.txt')

        #Use model to predict
        preds_x = modelx.predict(data.iloc[:, :-1])
        preds_y = modely.predict(data.iloc[:, :-1])
        preds_f = modelf.predict(data.iloc[:, :-1]).astype(int)

        #Prepare all prediction data
        test_preds = pd.DataFrame(np.stack((preds_f, preds_x, preds_y))).T
        test_preds.columns = subm.columns
        test_preds.index = data["site_path_timestamp"]
        test_preds["floor"] = test_preds["floor"].astype(int)
        predictions.append(test_preds)

        print(f"Finished site {n_files}")

    return predictions

if __name__ == '__main__':
    
    #Create Models
    create_models()

    #Make predictions
    all_preds = predict()

    #Crate submission file
    create_submission_file(all_preds)