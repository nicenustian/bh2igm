import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from machine_learning_models.MLP import MLP
from tensorflow.keras import Input, Model
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers

class ConvLayer(keras.Model):
    def __init__(self, filters, seed, kernel=3, stride=1, scale_factor=2, 
                 pooling_int=1, name="ConvLayer", **kwargs):
        """
        Class representing a single Convolutional Layer.

        Args:
            filters (int): Number of filters for the convolutional layer.
            seed (int): Seed for random initializations.
            kernel (int): Size of the convolutional kernel.
            stride (int): Stride for the convolution operation.
            scale_factor (int): Scale factor for pooling.
            pooling_int (int): Pooling type indicator. 1 for MaxPooling, 2 for AveragePooling, else Upsampling.
            name (str): Name of the layer.
        """
        super(ConvLayer, self).__init__(name=name, **kwargs)

        activation = tfkl.PReLU(alpha_initializer=tf.initializers.constant(0.3))
        self.conv = tfkl.Conv1D(filters, kernel, stride, padding='same')
        self.bn = tfkl.BatchNormalization(name='bn')
        self.act = tfkl.Activation(activation, name='act')

        if pooling_int == 1:
            self.scale = tfkl.MaxPooling1D(scale_factor, name='maxpool')
        elif pooling_int == 2:
            self.scale = tfkl.AveragePooling1D(scale_factor, name='avgpool')
        else:
            self.scale = tfkl.UpSampling1D(scale_factor, name='upsample')

    def call(self, inputs):
        """
        Perform forward pass through the Convolutional Layer.

        Args:
            inputs: Input tensor for the layer.

        Returns:
            Output tensor after convolution, batch normalization, activation, and pooling.
        """
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.scale(x)

        return x


class ConvNet(keras.Model):
    def __init__(self, num_of_blocks=[2, 2], num_of_channels=[4, 8], nodes=1024, seed=12345, name="ConvNet", **kwargs):
        """
        Class representing a Convolutional Neural Network.

        Args:
            num_of_blocks (list): Number of blocks in the network.
            num_of_channels (list): Number of channels for each block.
            nodes (int): Number of nodes in the Dense layer.
            seed (int): Seed for random initializations.
            name (str): Name of the network.
        """
        super(ConvNet, self).__init__(name=name, **kwargs)

        self.convnet_layers = []

        for block_index, layers_in_block in enumerate(num_of_blocks):
            for layer_num_in_block in range(layers_in_block):
                scale_factor = (2 if layer_num_in_block == 0 else 1)
                self.convnet_layers.append(ConvLayer(num_of_channels[block_index], seed, 3, 1, scale_factor))

        self.flat = tfkl.Flatten()
        self.dense = MLP(nodes * 2, seed, name='MLP1')
        self.prob = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :nodes], scale=1e-5 + tf.math.softplus(t[..., nodes:])), name='dist')

  
    def call(self, inputs):
        """
        Perform forward pass through the Convolutional Neural Network.

        Args:
            inputs: Input tensor for the network.

        Returns:
            Output tensor after passing through the network layers.
        """
        x = inputs
        for li, layer in enumerate(self.convnet_layers.layers):
            layer._name = 'ConvLayer' + str(li + 1)
            x = layer((x))

        x = self.flat(x)
        x = self.dense(x)
        x = self.prob(x)

        return x
    
    
    def summary(self, shape):
        """
        Generate a summary of the ResNet model architecture.

        Args:
            shape: Shape of the input.

        Returns:
            Summary of the model architecture.
        """
        x = Input(shape=shape)
        model = Model(inputs=x, outputs=self.call(x))
        return model.summary()


#model = ConvNet([2, 2], [8, 16], 1024, 12345)
#model.summary((1024, 2))