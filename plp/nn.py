import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import sys
print(tf.__version__)
np.set_printoptions(threshold=sys.maxsize)

from helpers.get_data import data


[
    [
        X_train,
        y_train
    ],
    [
        X_test,
        y_test
    ]
] = data.get()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)


input_layer = tf.keras.layers.Input(shape=(X_train.shape[1],))
dense_layer_1 = tf.keras.layers.Dense(300, activation='relu')(input_layer)
dense_layer_2 = tf.keras.layers.Dense(300, activation='relu')(dense_layer_1)
dense_layer_3 = tf.keras.layers.Dense(300, activation='relu')(dense_layer_2)
dense_layer_4 = tf.keras.layers.Dense(300, activation='relu')(dense_layer_3)
dense_layer_5 = tf.keras.layers.Dense(300, activation='relu')(dense_layer_4)
dense_layer_6 = tf.keras.layers.Dense(300, activation='relu')(dense_layer_5)
dense_layer_7 = tf.keras.layers.Dense(300, activation='relu')(dense_layer_6)
dense_layer_8 = tf.keras.layers.Dense(300, activation='relu')(dense_layer_7)
output = tf.keras.layers.Dense(1)(dense_layer_8)

model = tf.keras.models.Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10000)

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)