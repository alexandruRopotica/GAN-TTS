import tensorflow as tf
from Models.GeneratorNet.condBatchNorm import ConditionalBatchNorm
from Utils.utils import SpectralConv1D, SpectralConv1DTranspose


class GeneratorBlock(tf.keras.Model):
    def __init__(self, channels, isTraining, upsampleFactor=1, **kwargs):
        super(GeneratorBlock, self).__init__(**kwargs)
        self.channels = channels
        self.upsampleFactor = upsampleFactor
        self.isTraining = isTraining
        self.firstCBN = ConditionalBatchNorm(self.isTraining)
        self.firstStack = tf.keras.Sequential([
            SpectralConv1DTranspose(self.channels, 3, strides=self.upsampleFactor),
            SpectralConv1D(self.channels, 3)])
        self.secondCBN = ConditionalBatchNorm(self.isTraining)
        self.firstDilatedConv = SpectralConv1D(self.channels, 3, dilation=2)
        self.residualStack = tf.keras.Sequential([
            SpectralConv1DTranspose(self.channels, 3, strides=self.upsampleFactor),
            SpectralConv1D(self.channels, 1)])
        self.thirdCBN = ConditionalBatchNorm(self.isTraining)
        self.secondDilatedConv = SpectralConv1D(self.channels, 3, dilation=4)
        self.fourthCBN = ConditionalBatchNorm(self.isTraining)
        self.finalDilatedConv = SpectralConv1D(self.channels, 3, dilation=8)
    

    def call(self, inputs, noise):
        outputs = self.firstCBN(inputs, noise)
        outputs = self.firstStack(outputs)
        outputs = self.secondCBN(outputs, noise)
        outputs = self.firstDilatedConv(outputs)
        residualOutputs = self.residualStack(inputs)
        outputs = outputs + residualOutputs
        outputs = self.thirdCBN(outputs, noise)
        outputs = self.secondDilatedConv(outputs)
        outputs = self.fourthCBN(outputs, noise)
        outputs = self.finalDilatedConv(outputs)
        return outputs