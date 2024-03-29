import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_d):
    w1 = int(input_d[0]/4)
    w2 = int(input_d[0]/2)
    w3 = int(input_d[0])
    model = keras.Sequential()
    model.add(layers.Dense(w3, activation='relu', input_shape=(input_d[1],)))
    model.add(layers.Dense(w2, activation='relu'))
    model.add(layers.Dense(w2, activation='relu'))
    model.add(layers.Dense(w2, activation='relu'))
    model.add(layers.Dense(w1, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.optimizers.Adam(lr=0.001),
                  loss='mse', 
                  metrics=['mean_squared_error',keras.metrics.RootMeanSquaredError()])
    return model