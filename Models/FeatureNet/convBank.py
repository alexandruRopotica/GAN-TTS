import tensorflow as tf


class Conv1DBank(tf.keras.Model):
    def __init__(self, channels, kernelSize, activation, isTraining, **kwargs):
        super(Conv1DBank, self).__init__(**kwargs)
        self.channels = channels
        self.kernelSize = kernelSize
        self.activation = activation
        self.isTraining = isTraining
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.channels, kernel_size=self.kernelSize,
            activation=self.activation, padding='same')
        self.batchNorm = tf.keras.layers.BatchNormalization(trainable=self.isTraining)

    def call(self, inputs):
        outputs = self.conv1d(inputs)
        outputs = self.batchNorm(outputs)
        return outputs