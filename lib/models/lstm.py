import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, num_classes=128):
        super(Model, self).__init__(name='LSTM')

        self.num_classes = num_classes
        self.lstm = tf.keras.layers.LSTM(num_classes)
    
    def call(self, inputs):
        x = self.lstm(inputs)
        return x
