import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow_text as text
from Models.bert import BERT
from Models.FeatureNet.cbhg import CBHG
from Models.GeneratorNet.generator import Generator
from Models.DiscriminatorNet.discriminator import Discriminator
from Training.train import getSamples


PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
ENCODER = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1"
TEXT_INPUT = ['This is such an amazing movie!']
BATCH_SIZE = 1
NOISE = tf.random.normal((BATCH_SIZE, 128, 1))
WINDOWS = [240, 480, 960, 1920, 3600]
BERT_MODEL = BERT(PREPROCESSOR, ENCODER)


def testBERT():
    output = BERT_MODEL(TEXT_INPUT)
    assert output.shape == (1,256,1)
    print("BERT test completed.")


def testFeatureNet():
    embeddings = BERT_MODEL(TEXT_INPUT)
    featureNet = CBHG(BATCH_SIZE, 16, True)
    genFeatures, discFeatures = featureNet(embeddings)
    assert genFeatures.shape == (1, 400, 256)
    assert discFeatures.shape == (1, 1, 256)
    print("FeatureNet test completed.")


def testGeneratorNet():
    embeddings = BERT_MODEL(TEXT_INPUT)
    featureNet = CBHG(BATCH_SIZE, 16, True)
    genFeatures, _ = featureNet(embeddings)
    generator = Generator(BATCH_SIZE, True)
    generatedAudio = generator(genFeatures, NOISE)
    assert generatedAudio.shape == (1, 48000, 1)
    print("GeneratorNet test completed.")


def testDiscriminatorNet():
    embeddings = BERT_MODEL(TEXT_INPUT)
    featureNet = CBHG(BATCH_SIZE, 16, True)
    genFeatures, discFeatures = featureNet(embeddings)
    generator = Generator(BATCH_SIZE, True)
    discriminator = Discriminator()
    generatedAudio = generator(genFeatures, NOISE)
    w1, w2, w3, w4, w5 = getSamples(generatedAudio, WINDOWS)
    fakeAudioPred = discriminator(w1, w2, w3, w4, w5, discFeatures)
    assert fakeAudioPred.shape == (1, 1)
    print("DiscriminatorNet test completed.")


if __name__ == '__main__':
    with tf.device('device:GPU:0'):
        testBERT()
        testFeatureNet()
        testGeneratorNet()
        testDiscriminatorNet()
