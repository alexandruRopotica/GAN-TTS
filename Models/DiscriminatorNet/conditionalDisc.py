import tensorflow as tf
from Utils.utils import SpectralConv1D
from Model.DiscriminatorNet.conditionalDBlock import ConditionalDBlock
from Model.DiscriminatorNet.discriminatorBlock import DiscriminatorBlock


class ConditionalDiscriminator(tf.keras.Model):
    def __init__(self, downsampleFactor, factors, **kwargs):
        super(ConditionalDiscriminator, self).__init__(**kwargs)
        self.downsampleFactor = downsampleFactor
        self.factors = factors
        dblockList = []
        dblockSize = 64
        self.reshape = tf.keras.Sequential([
            SpectralConv1D(filters=self.downsampleFactor, kernelSize=1),
            tf.keras.layers.MaxPool1D(self.downsampleFactor, padding='same')])
        for i in range(len(self.factors) - 1):
            dblockList.append(DiscriminatorBlock(dblockSize, self.factors[i]))
            dblockSize = dblockSize * 2
        self.dblockStack = tf.keras.Sequential(dblockList)
        self.condDBlock = ConditionalDBlock(dblockSize, self.factors[-1])
        self.finalDBlocks = tf.keras.Sequential([
            DiscriminatorBlock(dblockSize, 1),
            DiscriminatorBlock(dblockSize, 1)])
        self.avgPool = tf.keras.layers.AveragePooling1D()
        
    def call(self, inputs, condition):
        outputs = self.reshape(inputs)
        outputs = self.dblockStack(outputs)
        outputs = self.condDBlock(outputs, condition)
        outputs = self.finalDBlocks(outputs)
        outputs = self.avgPool(outputs)
        return outputs