import tensorflow as tf
from Utils.utils import SpectralConv1D


class ConditionalDBlock(tf.keras.Model):
    def __init__(self, filters, downsampleFactor, **kwargs):
        super(ConditionalDBlock, self).__init__(**kwargs)
        self.filters = filters
        self.downsampleFactor = downsampleFactor
        self.firstStack = tf.keras.Sequential([
            tf.keras.layers.MaxPool1D(self.downsampleFactor, padding='same'),
            tf.keras.layers.ReLU(),
            SpectralConv1D(filters=self.filters, kernelSize=3)])
        self.featureConv = SpectralConv1D(filters=self.filters, kernelSize=1)
        self.secondStack = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            SpectralConv1D(filters=self.filters, kernelSize=3, dilation=2)])
        self.residualStack = tf.keras.Sequential([
            SpectralConv1D(filters=self.filters, kernelSize=1),
            tf.keras.layers.MaxPool1D(self.downsampleFactor, padding='same')])

    def call(self, inputs, condition):
        outputs = self.firstStack(inputs)
        featureOutputs = self.featureConv(condition)
        outputs = outputs + featureOutputs
        outputs = self.secondStack(outputs)
        residualOutputs = self.residualStack(inputs)
        outputs = outputs + residualOutputs
        return outputs