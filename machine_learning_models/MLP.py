import tensorflow as tf
from tensorflow.keras import layers
tfkl = tf.keras.layers

class MLP(layers.Layer):

    def __init__(self, nodes, seed, name="MLP", **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)
        
        self.nodes =  nodes
        self.seed = seed
        activation = tfkl.PReLU(alpha_initializer=tf.initializers.constant(0.3))

        self.dense = tfkl.Dense(nodes, name='dense')
        self.bn = tfkl.BatchNormalization(name = 'bn')
        self.act = tfkl.Activation(activation, name = 'act')
      
    def call(self, inputs):
        x = self.dense(inputs)
        return self.act(self.bn(x))