import tensorflow as tf
from Models.FeatureNet.convBank import Conv1DBank


class CBHG(tf.keras.Model):
    def __init__(self, batchSize, K, isTraining, **kwargs):
        super(CBHG, self).__init__(**kwargs)
        self.batchSize = batchSize
        self.K = K
        self.isTraining = isTraining
        self.ConvBanks = [Conv1DBank(128, i, tf.nn.relu, self.isTraining) for i in range(1, self.K + 1)]
        self.maxPooling = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding='same')
        self.firstProjectionConv = Conv1DBank(128, 3, tf.nn.relu, self.isTraining)
        self.secondProjectionConv = Conv1DBank(128, 3, None, self.isTraining)
        self.highwayNet = tf.keras.Sequential([tf.keras.layers.Dense(128, tf.nn.relu) for i in range(4)])
        self.bidirectionalGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(64, return_sequences=True), 
            backward_layer=tf.keras.layers.GRU(64, return_sequences=True, go_backwards=True))
        self.encoderPreNet = tf.keras.Sequential([
            tf.keras.layers.Dense(256, tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, tf.nn.relu),
            tf.keras.layers.Dropout(0.5)])
        self.lastProjectionConv = Conv1DBank(1, 3, None, self.isTraining)
        self.upsample = tf.keras.layers.UpSampling1D(size=400)
        self.conv = tf.keras.layers.Conv1D(256, 3, padding='same')
    
    def call(self, inputs):
        outputList = []
        for convBank in self.ConvBanks:
            outputList.append(convBank(inputs))
        outputs = tf.keras.layers.concatenate(outputList)
        outputs = self.maxPooling(outputs)
        outputs = self.firstProjectionConv(outputs)
        outputs = self.secondProjectionConv(outputs)
        highwayOutputs = outputs + inputs
        outputs = self.highwayNet(highwayOutputs)
        outputs = self.bidirectionalGRU(outputs)
        outputs = self.encoderPreNet(outputs)
        outputs = self.lastProjectionConv(outputs)
        outputs = tf.reshape(outputs, (self.batchSize, 1, 256))
        discOutputs = outputs
        outputs = self.upsample(outputs)
        genOutputs = self.conv(outputs)
        return genOutputs, discOutputs