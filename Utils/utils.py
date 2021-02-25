import tensorflow as tf
import tensorflow_addons as tfa


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, beta=1e-4, **kwargs):
        super(OrthogonalRegularizer, self).__init__(**kwargs)
        self.beta = beta

    def call(self, input_tensor):
        c = input_tensor.shape[-1]
        x = tf.reshape(input_tensor, (-1, c))
        ortho_loss = tf.matmul(x, x, transpose_a=True) * (1 - tf.eye(c))
        outputs = self.beta * tf.norm(ortho_loss)
        return outputs


class SpectralConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernelSize, strides=1,
                padding='same', dilation=1, activation=None,
                kernelInit=tf.initializers.Orthogonal,
                kernelReg=OrthogonalRegularizer(), **kwargs):
        super(SpectralConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.activation = activation
        self.kernelInit = kernelInit
        self.kernelReg = kernelReg
        self.spectralConv = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv1D(
                filters=self.filters, kernel_size=self.kernelSize, strides=self.strides,
                padding=self.padding, dilation_rate=self.dilation, activation=self.activation,
                kernel_initializer=self.kernelInit, kernel_regularizer=self.kernelReg))
  
    def call(self, inputs):
        outputs = self.spectralConv(inputs)
        return outputs


class SpectralConv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernelSize, strides, padding='same',
                kernelInit=tf.initializers.Orthogonal,
                kernelReg=OrthogonalRegularizer(), **kwargs):
        super(SpectralConv1DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
        self.padding = padding
        self.kernelInit = kernelInit
        self.kernelReg = kernelReg
        self.spectralConvTranspose = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv1DTranspose(
                filters=self.filters, kernel_size=self.kernelSize,
                strides=self.strides, padding=self.padding,
                kernel_initializer=self.kernelInit, kernel_regularizer=self.kernelReg))
  
    def call(self, inputs):
        outputs = self.spectralConvTranspose(inputs)
        return outputs