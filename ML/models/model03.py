import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_d):
    w1 = int(input_d/2)
    w2 = int(input_d/20)
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape(input_d,1)))
    model.add(layers.Dense(w1, activation='sigmoid'))
    model.add(layers.Dense(w2, activation='sigmoid'))
    model.add(layers.Dense(3, activation='sigmoid'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, 
                        beta_2=0.999, epsilon=1e-07, 
                        amsgrad=False,name='Adam'),
                  loss='mse', 
                  metrics=['mean_squared_error'])
    #model.summary()
    return model