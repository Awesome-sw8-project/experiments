import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(shp):
    model = keras.Sequential()
    model.add(layers.Masking(mask_value=0.,input_shape=(shp[1], shp[2])))
    model.add(layers.LSTM(round(shp[2]/4), return_sequences=True, activation="relu"))
    model.add(layers.LSTM(1, return_sequences=True))
    model.compile(optimizer=tf.optimizers.Adam(lr=0.001),
                  loss='mse', 
                  metrics=['mean_squared_error',keras.metrics.RootMeanSquaredError()])
    return model