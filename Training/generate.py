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
import numpy as np


DATASET_DIR = './LJspeechTest'
CKPT = './Checkpoints/'


def FrechetInceptionDistance(realAudio, generatedAudio):
    inceptionv3 = tf.keras.applications.InceptionV3(include_top=False, input_shape=(128,125,3), pooling='avg')
    realFeatures = inceptionv3(realAudio)
    generatedFeatures = inceptionv3(generatedAudio)
    muReal, sigmaReal = np.mean(realFeatures), np.cov(realFeatures, rowvar=False)
    muGenerated, sigmaGenerated = np.mean(generatedFeatures), np.cov(generatedFeatures, rowvar=False)
    muDiff = np.linalg.norm(muReal- muGenerated)
    covDiff = sigmaReal + sigmaGenerated - 2*(tf.math.sqrt(sigmaReal*sigmaGenerated))
    fid = muDiff**2 + np.trace(covDiff)
    return fid


def saveGeneratedAudio():
    checkpoint = tf.train.Checkpoint(genOptimizer=genOptimizer,
                                 discOptimizer=discOptimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 featureNet=featureNet)
    models = checkpoint.restore(os.path.join(CKPT, 'ckpt-1.data-00000-of-00001'))
    print(models.generator.summary())


if __name__=='__main__':
    saveGeneratedAudio()