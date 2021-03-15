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


DATASET_DIR = './LJspeechTest'
CKPT = './Checkpoints/'



def saveGeneratedAudio():
    genOptimizer, discOptimizer = None
    generator = Generator(10, False)
    featureNet = CBHG(BATCH_SIZE, 10, True, 768)
    dicriminator = Discriminator()
    checkpoint = tf.train.Checkpoint(genOptimizer=genOptimizer,
                                 discOptimizer=discOptimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 featureNet=featureNet)
    
    checkpoint.restore(os.path.join(CKPT, 'ckpt-1.data-00000-of-00001'))
    print(generator.summary)