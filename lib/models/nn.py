import tensorflow as tf
from tensorflow.keras import layers


class NN(tf.keras.Model):
    def __init__(self, num_classes=128):
        super(NN, self).__init__(name='NN')
        self.num_classes = num_classes
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = tf.math.l2_normalize(x)
        return x
