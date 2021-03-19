import numpy as np
import tensorflow as tf

class LSTMClassifier():

    def __init__(self, n_classes=2, timesteps=7, n_features=2, activation='tanh', metrics=['accuracy']):
        self.n_classes = n_classes
        self.timesteps = timesteps
        self.n_features = n_features
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.timesteps, self.n_features)),
            tf.keras.layers.LSTM(n_classes, activation=activation)
            ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=metrics)

    def fit(self, x, y, epochs=10):
        self.model.fit(x, y, epochs=epochs)

    def predict(self, x):
        logits = self.model.predict(x)
        return np.argmax(logits, 1)

    def scores(self, x):
        logits = self.model.predict(x)
        scores = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        return scores.numpy()
