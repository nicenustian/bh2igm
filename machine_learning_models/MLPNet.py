import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from machine_learning_models.MLP import MLP
from tensorflow.keras import Input, Model
import numpy as np
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers


class MLPNet(keras.Model):

    def __init__(self, num_of_layers=[2], nodes=1024, seed=12345, 
                 name="MLPNet", **kwargs):
        super(MLPNet, self).__init__(name=name, **kwargs)

        self.mlpnet_layers = []
        self.num_of_layers = num_of_layers
        self.nodes = nodes
        activation = tf.keras.layers.PReLU(alpha_initializer=
                                           tf.initializers.constant(0.3))


        for li in range(np.sum(num_of_layers)):
                
            self.mlpnet_layers.append(
                tfkl.Dense(nodes*2, name='dense'+str(li+1)))
            self.mlpnet_layers.append(
                tfkl.BatchNormalization(name = 'bn'+str(li+1)))
            self.mlpnet_layers.append(
                tfkl.Activation(activation, name = 'act'+str(li+1)))

        self.flat = tfkl.Flatten()

        self.prob = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :nodes], scale=1e-5 + \
                                 tf.math.softplus(t[..., nodes:])), name='dist')

          

    def call(self, inputs):
        x = inputs
        x = self.flat(x)
        
        for li, layer in enumerate(self.mlpnet_layers.layers):
            layer._name = 'MLPLayer'+str(li+1)
            x = layer(x)
        x = self.prob(x)

        return x
