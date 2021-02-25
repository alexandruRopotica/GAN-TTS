import tensorflow as tf
from generatorBlock import GeneratorBlock
from utils import SpectralConv1D


class Generator(tf.keras.Model):
    def __init__(self, batchSize, isTraining,  **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.batchSize = batchSize
        self.isTraining = isTraining
        self.preProcess = SpectralConv1D(filters=768, kernelSize=3)
        self.generatorBlocks = [
            GeneratorBlock(768, self.isTraining, 1),
            GeneratorBlock(768, self.isTraining, 1),
            GeneratorBlock(768, self.isTraining, 2),
            GeneratorBlock(384, self.isTraining, 2),
            GeneratorBlock(384, self.isTraining, 2),
            GeneratorBlock(384, self.isTraining, 3),
            GeneratorBlock(192, self.isTraining, 5)]
        self.postProcess = SpectralConv1D(filters=1, kernelSize=3, activation='tanh')

    def call(self, inputs, noise):
        outputs = self.preProcess(inputs)
        for gblock in self.generatorBlocks:
            outputs = gblock(outputs, noise)
        outputs = self.postProcess(outputs)
        outputs = tf.reshape(outputs, shape=(self.batchSize, 48000, 1))
        return outputs