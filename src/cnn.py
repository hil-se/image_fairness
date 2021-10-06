import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self, start_size = 4):
        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(200, 200, 3)))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size*4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(25*25*start_size*4, activation='relu'))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'], optimizer='adam')


    def fit(self, X, y, sample_weight=None):
        y_hot =tf.keras.utils.to_categorical(y, 2)
        self.model.fit(X, y_hot, sample_weight=sample_weight)

    def predict(self, X):
        pred = self.model.predict(X)
        return np.array([np.argmax(row) for row in pred])
