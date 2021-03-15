import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from transformers import logging
logging.set_verbosity_error()
from Models.bert import BERT
from Models.FeatureNet.cbhg import CBHG
from Models.GeneratorNet.generator import Generator
from Models.DiscriminatorNet.discriminator import Discriminator
from Training.train import getSamples


TEXT_INPUT = ['This is such an amazing movie!']
BATCH_SIZE = 1
NOISE = tf.random.normal((BATCH_SIZE, 128, 1))
WINDOWS = [240, 480, 960, 1920, 3600]
BERT_TYPE = 'bert-base-cased'
BERT_MODEL = BERT(BERT_TYPE)


def testBERT():
    output = BERT_MODEL(TEXT_INPUT)
    assert output.shape == (1,768,1)
    print("BERT test completed.")


def testFeatureNet():
    embeddings = BERT_MODEL(TEXT_INPUT)
    featureNet = CBHG(BATCH_SIZE, 16, True, 768)
    genFeatures, discFeatures = featureNet(embeddings)
    assert genFeatures.shape == (1, 400, 768)
    assert discFeatures.shape == (1, 1, 768)
    print("FeatureNet test completed.")


def testGeneratorNet():
    embeddings = BERT_MODEL(TEXT_INPUT)
    featureNet = CBHG(BATCH_SIZE, 16, True, 768)
    genFeatures, _ = featureNet(embeddings)
    generator = Generator(BATCH_SIZE, True)
    generatedAudio = generator(genFeatures, NOISE)
    assert generatedAudio.shape == (1, 48000, 1)
    print("GeneratorNet test completed.")


def testDiscriminatorNet():
    embeddings = BERT_MODEL(TEXT_INPUT)
    featureNet = CBHG(BATCH_SIZE, 16, True, 768)
    genFeatures, discFeatures = featureNet(embeddings)
    generator = Generator(BATCH_SIZE, True)
    discriminator = Discriminator()
    generatedAudio = generator(genFeatures, NOISE)
    w1, w2, w3, w4, w5 = getSamples(generatedAudio, WINDOWS)
    fakeAudioPred = discriminator(w1, w2, w3, w4, w5, discFeatures)
    assert fakeAudioPred.shape == (1, 1)
    print("DiscriminatorNet test completed.")


if __name__ == '__main__':
    testBERT()
    testFeatureNet()
    testGeneratorNet()
    testDiscriminatorNet()
