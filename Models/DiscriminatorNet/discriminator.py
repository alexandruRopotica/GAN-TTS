import tensorflow as tf
from Model.DiscriminatorNet.unconditionalDisc import UnconditionalDiscriminator
from Model.DiscriminatorNet.conditionalDisc import ConditionalDiscriminator


class Discriminator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.uDscriminatorStack = [
            UnconditionalDiscriminator(1, (5, 3)),
            UnconditionalDiscriminator(2, (5, 3)),
            UnconditionalDiscriminator(4, (5, 3)),
            UnconditionalDiscriminator(8, (5, 3)),
            UnconditionalDiscriminator(15, (2, 2))]
        self.cDiscriminatorStack = [
            ConditionalDiscriminator(1, (1, 5, 3, 2, 2, 2)),
            ConditionalDiscriminator(2, (1, 5, 3, 2, 2)),
            ConditionalDiscriminator(4, (1, 5, 3, 2, 2)),
            ConditionalDiscriminator(8, (1, 5, 3)),
            ConditionalDiscriminator(15, (1, 2, 2, 2))  
        ]
        self.flatten = tf.keras.layers.Flatten()
        self.denseStack = ([tf.keras.layers.Dense(1) for i in range(5)])
    
        
    def call(self, w1Inputs, w2Inputs, w3Inputs, w4Inputs, w5Inputs, condition):
        outputs = 0
        windows = [w1Inputs, w2Inputs, w3Inputs, w4Inputs, w5Inputs]
        for uDisc, cDisc, window, dense in zip(self.uDscriminatorStack, self.cDiscriminatorStack, windows, self.denseStack):
            outputs += dense(self.flatten(uDisc(window)) + self.flatten(cDisc(window, condition)))
        return outputs