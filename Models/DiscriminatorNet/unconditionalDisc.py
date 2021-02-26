import tensorflow as tf
from utils import SpectralConv1D
from DiscriminatorNet.discriminatorBlock import DiscriminatorBlock


class UnconditionalDiscriminator(tf.keras.Model):
    def __init__(self, downsampleFactor, factors, **kwargs):
        super(UnconditionalDiscriminator, self).__init__(**kwargs)
        self.downsampleFactor = downsampleFactor
        self.factors = factors
        self.reshapeNet = tf.keras.Sequential([
            SpectralConv1D(filters=self.downsampleFactor, kernelSize=1),
            tf.keras.layers.MaxPool1D(self.downsampleFactor, padding='same')
            ])
        self.dBlockStack = tf.keras.Sequential([
            DiscriminatorBlock(64, 1),
            DiscriminatorBlock(128, self.factors[0]),
            DiscriminatorBlock(256, self.factors[1]),
            DiscriminatorBlock(256, 1),
            DiscriminatorBlock(256, 1)
        ])
        self.avgPool = tf.keras.layers.AveragePooling1D()
        
    def call(self, inputs):
        outputs = self.reshapeNet(inputs)
        outputs = self.dBlockStack(outputs)
        outputs = self.avgPool(outputs)
        return outputs