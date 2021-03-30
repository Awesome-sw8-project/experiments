import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_d):
    w1 = int(input_d/2)
    w2 = int(input_d/4)
    model = keras.Sequential()
    model.add(layers.Dense(w1, activation='relu'))
    model.add(layers.Dense(w2, activation='sigmoid'))
    model.add(layers.Dense(w2, activation='sigmoid'))
    model.add(layers.Dense(3))
    model.compile(optimizer=tf.optimizers.Adam(lr=0.01),
                  loss='mse', 
                  metrics=['mean_squared_error'])
    #model.summary()
    return model