import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from mlp import MLP
from tensorflow.keras import Input, Model
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers


def pad(x, padding=1):
    """
    Periodic padding for signals.
    Works for larger scale inputs skewers, not cropped small scale sections.

    Args:
        x (tensor): Input tensor to be padded.
        padding (int): Padding size for the input tensor.

    Returns:
        Padded input tensor.
    """
    return tf.concat([x[:, -padding:, :], x, x[:, :padding, :]], axis=1)


class Residual(tf.keras.Model):
    """
    The Residual block of ResNet.
    """

    def __init__(self, num_channels, seed, use_1x1conv=False, strides=1, name="Residual"):
        super().__init__()

        self.conv1 = tfkl.Conv1D(num_channels, 3, strides, padding='valid')
        self.conv2 = tfkl.Conv1D(num_channels, 3, 1, padding='valid')
        self.conv3 = None

        if use_1x1conv:
            self.conv3 = tfkl.Conv1D(num_channels, 1, strides, padding='valid')

        activation = tfkl.PReLU(alpha_initializer=tf.initializers.constant(0.3))
        self.bn1 = tfkl.BatchNormalization(name='bn1')
        self.bn2 = tfkl.BatchNormalization(name='bn2')
        self.act1 = tfkl.Activation(activation, name='act1')
        self.act2 = tfkl.Activation(activation, name='act2')

    def call(self, X):
        Y = self.act1(self.bn1(self.conv1(pad(X))))
        Y = self.bn2(self.conv2(pad(Y)))

        if self.conv3 is not None:
            X = self.conv3(X)
        return self.act2(X + Y)

    def summary(self, shape):
        """
        Generate a summary of the Residual model architecture.

        Args:
            shape: Shape of the input.

        Returns:
            Summary of the model architecture.
        """
        x = Input(shape=shape)
        model = Model(inputs=x, outputs=self.call(x))
        return model.summary()


class ResnetBlock(keras.Model):
    def __init__(self, num_channels, num_residuals, seed, name="ResBlock", **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []

        for i in range(num_residuals):
            if i == 0:
                self.residual_layers.append(
                    Residual(num_channels, seed, use_1x1conv=True,
                             strides=2, name='Residual' + str(i + 1)))
            else:
                self.residual_layers.append(Residual(num_channels,
                                                     seed, use_1x1conv=False,
                                                     strides=1))

        self.scale = tfkl.MaxPooling1D(1)

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return self.scale(X)

    def summary(self, shape):
        x = Input(shape=shape)
        model = Model(inputs=x, outputs=self.call(x))
        return model.summary()


class ResNet(keras.Model):
    def __init__(self, num_of_blocks, num_of_channels, nodes, seed, name="ResNet", **kwargs):
        """
        ResNet model consisting of residual blocks.

        Args:
            num_of_blocks (list): Number of blocks.
            num_of_channels (list): Number of channels for each block.
            nodes (int): Number of nodes.
            seed (int): Seed for random initialization.
            name (str): Name of the model.
        """
        super(ResNet, self).__init__(name=name, **kwargs)

        # Initialize ResNet components
        self.nodes = nodes
        self.residual_layers = []

        # Create residual blocks for the model
        for i in range(len(num_of_blocks)):
            self.residual_layers.append(ResnetBlock(num_of_channels[i],
                                                     num_of_blocks[i], seed,
                                                     name='ResBlock' + str(i + 1)))

        # Additional layers
        self.flat = tfkl.Flatten()
        self.dense1 = MLP(self.nodes * 2, seed, name='MLP1')

        # Probabilistic distribution layer
        self.prob = tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :nodes],
                                                                 scale=1e-5 + tf.math.softplus(t[..., nodes:])),
                                            name='dist')

    def call(self, inputs, training=True):
        """
        Define the forward pass through the ResNet model.

        Args:
            inputs: Input data.
            training (bool): Indicates whether the model is in training mode.

        Returns:
            Output tensor from the model.
        """
        x = inputs

        # Pass through each residual block in the model
        for layer in self.residual_layers.layers:
            x = layer(x)

        # Additional layers
        x = self.flat(x)
        x = self.dense1(x)
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

    def __init__(self, num_of_blocks, num_of_channels, nodes, seed, name="ResNet", **kwargs):
        super(ResNet, self).__init__(name=name, **kwargs)

        self.nodes = nodes
        self.residual_layers = []
        for i in range(len(num_of_blocks)):
            self.residual_layers.append(ResnetBlock(num_of_channels[i],
                                                     num_of_blocks[i], seed,
                                                     name='ResBlock' + str(i + 1)))

        self.flat = tfkl.Flatten()
        self.dense1 = MLP(self.nodes * 2, seed, name='MLP1')
        self.prob = tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :nodes],
                                                                 scale=1e-5 + tf.math.softplus(t[..., nodes:])), name='dist')

    def call(self, inputs, training=True):
        x = inputs
        for layer in self.residual_layers.layers:
            x = layer(x)

        x = self.flat(x)
        x = self.dense1(x)
        x = self.prob(x)

        return x

    def summary(self, shape):
        x = Input(shape=shape)
        model = Model(inputs=x, outputs=self.call(x))
        return model.summary()

    
#model = ResNet([2, 2], [8, 16], 1024, 12345)
#model.summary((1024, 2))