import tensorflow as tf
from Utils.utils import SpectralConv1D


class DiscriminatorBlock(tf.keras.Model):
    def __init__(self, filters, downsampleFactor, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        self.filters = filters
        self.downsampleFactor = downsampleFactor
        self.stack = tf.keras.Sequential([
            tf.keras.layers.MaxPool1D(self.downsampleFactor, padding='same'),
            tf.keras.layers.ReLU(),
            SpectralConv1D(filters=self.filters, kernelSize=3, activation=tf.nn.relu),
            SpectralConv1D(filters=self.filters, kernelSize=3, activation=tf.nn.relu, dilation=2)])
        self.residualStack = tf.keras.Sequential([
            SpectralConv1D(filters=self.filters, kernelSize=3),
            tf.keras.layers.MaxPool1D(self.downsampleFactor, padding='same')])
        
    def call(self, inputs):
        outputs = self.stack(inputs)
        residualOutputs = self.residualStack(inputs)
        outputs = residualOutputs + outputs
        return outputs