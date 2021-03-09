import tensorflow as tf
import tensorflow_addons as tfa


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, beta=1e-4, **kwargs):
        super(OrthogonalRegularizer, self).__init__(**kwargs)
        self.beta = beta

    def call(self, inputTensor):
        c = inputTensor.shape[-1]
        x = tf.reshape(inputTensor, (-1, c))
        orthoLoss = tf.matmul(x, x, transpose_a=True) * (1 - tf.eye(c))
        outputs = self.beta * tf.norm(orthoLoss)
        return outputs


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output
    
    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


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
        self.spectralConv = SpectralNormalization(
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
        self.spectralConvTranspose = SpectralNormalization(
            tf.keras.layers.Conv1DTranspose(
                filters=self.filters, kernel_size=self.kernelSize,
                strides=self.strides, padding=self.padding,
                kernel_initializer=self.kernelInit, kernel_regularizer=self.kernelReg))
  
    def call(self, inputs):
        outputs = self.spectralConvTranspose(inputs)
        return outputs