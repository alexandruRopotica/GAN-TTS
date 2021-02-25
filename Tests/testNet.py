import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow_text as text
from bert import BERT


def testBERT():
    PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    ENCODER = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1"
    textInput = ['This is such an amazing movie!']
    bert = BERT(PREPROCESSOR, ENCODER)
    output = bert(textInput)
    assert output.shape == (1,256,1)
    print("BERT test completed.")


if __name__ == '__main__':
    with tf.device('device:GPU:0'):
        testBERT()