import tensorflow as tf


class ConditionalBatchNorm(tf.keras.Model):
    def __init__(self, isTraining, units=1, **kwargs):
        super(ConditionalBatchNorm, self).__init__(**kwargs)
        self.units = units
        self.isTraining = isTraining
        self.randomIdx = np.random.randint(0, 128)
        self.instanceNorm = tfa.layers.InstanceNormalization()
        self.matrixGamma = tf.keras.layers.Dense(
            self.units, trainable=self.isTraining,
            kernel_initializer=tf.keras.initializers.Constant(1.0))
        self.matrixBeta = tf.keras.layers.Dense(
            self.units, trainable=self.isTraining,
            kernel_initializer=tf.keras.initializers.Constant(0.0))
        self.flatten = tf.keras.layers.Flatten()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, noise):
        outputs = self.instanceNorm(inputs)
        matrixGamma = self.flatten(self.matrixGamma(noise))
        matrixBeta = self.flatten(self.matrixBeta(noise))
        deltaGamma = matrixGamma[0][self.randomIdx]
        deltaBeta = matrixBeta[0][self.randomIdx]
        outputs = tf.multiply(deltaGamma, outputs) + deltaBeta
        outputs = self.relu(outputs)
        return outputs