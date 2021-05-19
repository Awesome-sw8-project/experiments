import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_d):
    w1 = int(input_d[0]/2)
    w2 = int(input_d[0]/4)
    model = keras.Sequential()
    #model.add(layers.Flatten(input_shape=input_d))
    #model.add(keras.layers.Flatten(input_shape=(input_d[0])))
    model.add(layers.Dense(w1, activation='sigmoid', input_shape=(input_d[0],)))
    model.add(layers.Dense(w2, activation='relu'))
    model.add(layers.Dense(w1, activation='sigmoid'))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.optimizers.Adam(lr=0.01),
                  loss='mse', 
                  metrics=['mean_squared_error'])
    #model.summary()
    return model