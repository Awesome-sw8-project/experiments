import lightgbm as lgb
import numpy as np
import joblib

class LightGBMWrapper:
    @staticmethod
    def predict(rssi, index):

        modelx = lgb.Booster(model_file=f'../../data/data/LightGBM_models/{index}modelx.txt')
        modely = lgb.Booster(model_file=f'../../data/data/LightGBM_models/{index}modely.txt')
        modelf = joblib.load(f'../../data/data/LightGBM_models/{index}modelf.txt')

        rssi = np.array(rssi).reshape((1,-1))
        preds_x = modelx.predict(rssi)
        preds_y = modely.predict(rssi)
        preds_f = modelf.predict(rssi).astype(int)

        return preds_x, preds_y, preds_f
