import tensorflow as tf
import tensorflow_hub as hub


class BERT(tf.keras.Model):
    def __init__(self, preprocessor, encoder, **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.preprocess = hub.KerasLayer(preprocessor)
        self.encode = hub.KerasLayer(encoder)

    def call(self, inputs):
        outputs = self.preprocess(inputs)
        outputs = self.encode(outputs)
        outputs = tf.expand_dims(outputs["pooled_output"], axis=-1)
        return outputs