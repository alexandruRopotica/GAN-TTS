import tensorflow as tf

class EmbeddingNet(tf.keras.Model):
    def __init__(self, textDataset, vocabSize=100):
        super(EmbeddingNet, self).__init__()
        self.vocabSize = vocabSize
        self.textDataset = textDataset
        self.encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(self.vocabSize)
        self.encoder.adapt(textDataset)
        self.embedder = tf.keras.layers.Embedding(self.vocabSize, 256)
        self.bidirectionalLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True), 
            backward_layer=tf.keras.layers.LSTM(128, return_sequences=True, go_backwards=True))
        self.poolingLayer = tf.keras.layers.GlobalAveragePooling1D()
    

    def call(self, inputs):
        outputs = self.encoder(inputs)
        outputs = self.embedder(outputs)
        outputs = self.bidirectionalLSTM(outputs)
        outputs = self.poolingLayer(outputs)
        outputs = tf.expand_dims(outputs, axis=-1)
        return outputs