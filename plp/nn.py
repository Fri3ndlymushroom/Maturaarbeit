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



input_layer = tf.keras.layers.Input(shape=(X_train.shape[1],))
dense_layer_1 = tf.keras.layers.Dense(250, activation='relu')(input_layer)
dense_layer_2 = tf.keras.layers.Dense(500, activation='relu')(dense_layer_1)
dense_layer_3 = tf.keras.layers.Dense(250, activation='relu')(dense_layer_2)
output =tf.keras.layers.Dense(1)(dense_layer_3)

model = tf.keras.models.Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_split=0.2)

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)


"""

model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(61, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=1000)

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)"""