import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


class BERT(tf.keras.Model):
    def __init__(self, bertType, **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(bertType)
        self.bert = TFBertModel.from_pretrained(bertType)

    def call(self, inputs):
        outputs = self.tokenizer(inputs, return_tensors='tf', padding='max_length')
        outputs = self.bert(outputs)
        outputs = tf.expand_dims(outputs["pooler_output"], axis=-1)
        return outputs