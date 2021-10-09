import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self, start_size = 4):
        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(200, 200, 3), use_bias=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
        # self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        # self.model.add(
        #     tf.keras.layers.Conv2D(start_size*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu', use_bias=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
        # self.model.add(tf.keras.layers.Dropout( 0.1))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        # self.model.add(
        #     tf.keras.layers.Conv2D(start_size*4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu', use_bias=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
        # self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        # self.model.add(tf.keras.layers.Dense(25*25*start_size*4, activation='relu', use_bias=True))
        self.model.add(tf.keras.layers.Dense(25 * 25 * start_size * 4, activation='relu', use_bias=True,
                                             kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'], optimizer='adam')


    def fit(self, X, y, sample_weight=None):
        y_hot =tf.keras.utils.to_categorical(y, 2)
        history = self.model.fit(X, y_hot, sample_weight=sample_weight, validation_split = 0.1, batch_size=128, epochs=10)
        print(history.history)

    def predict(self, X):
        pred = self.model.predict(X)
        return np.array([np.argmax(row) for row in pred])


class VGG:
    def __init__(self, start_size = 4):
        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                   input_shape=(200, 200, 3)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                   input_shape=(200, 200, 3)))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size*4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(6 * 6 * start_size * 8, activation='relu'))
        self.model.add(tf.keras.layers.Dense(6 * 6 * start_size * 8, activation='relu'))
        # self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'], optimizer='adam')


    def fit(self, X, y, sample_weight=None):
        y_hot =tf.keras.utils.to_categorical(y, 2)
        history = self.model.fit(X, y_hot, sample_weight=sample_weight, validation_split = 0.1, batch_size=128, epochs=10)
        print(history.history)

    def predict(self, X):
        pred = self.model.predict(X)
        return np.array([np.argmax(row) for row in pred])

class VGG16:
    def __init__(self):
        self.model = tf.keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(200,200,3),
            pooling=None,
            classes=2,
            classifier_activation="softmax",
        )
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'], optimizer='adam')


    def fit(self, X, y, sample_weight=None):
        y_hot =tf.keras.utils.to_categorical(y, 2)
        history = self.model.fit(X, y_hot, sample_weight=sample_weight, validation_split = 0.1, batch_size=128, epochs=10)
        print(history.history)

    def predict(self, X):
        pred = self.model.predict(X)
        return np.array([np.argmax(row) for row in pred])
