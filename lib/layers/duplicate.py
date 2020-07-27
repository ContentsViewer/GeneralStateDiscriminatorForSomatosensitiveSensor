import tensorflow as tf
from tensorflow.keras import layers

class Duplicate(layers.Layer):
    def __init__(self, **kwargs):
        super(Duplicate, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.concat([[inputs], [inputs]], 0)
