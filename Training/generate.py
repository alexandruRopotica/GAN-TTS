
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
import librosa
import soundfile as sf
from Training.train import getDataset
import tensorflow_hub as hub

TEXT_DIR = './LJSpeechTest/texts'
WAVS_DIR = './LJSpeechTest/wavs'
GENERATED_WAVS_DIR = './LJSpeechTest/generatedWavs'
BATCH_SIZE = 10
CKPT_DIR = './Checkpoints/'


def FrechetVGGDistance(realAudio, generatedAudio):
    vgg = hub.load('https://tfhub.dev/google/vggish/1')
    realFeatures, generatedFeatures = [], []
    for audio, genAudio in zip(realAudio, generatedAudio):
         features = vgg(audio)
         features = tf.reshape(features, (1, 384))
         realFeatures.append(features)
         features = vgg(genAudio)
         features = tf.reshape(features, (1, 384))
         generatedFeatures.append(features)
    realFeatures = tf.concat(realFeatures, axis=0).numpy()
    generatedFeatures = tf.concat(generatedFeatures, axis=0).numpy()
    muReal, sigmaReal = np.mean(realFeatures), np.cov(realFeatures, rowvar=False)
    muGenerated, sigmaGenerated = np.mean(generatedFeatures), np.cov(generatedFeatures, rowvar=False)
    muDiff = np.linalg.norm(muReal- muGenerated)
    covDiff = sigmaReal + sigmaGenerated - 2*(tf.math.sqrt(sigmaReal*sigmaGenerated))
    fid = muDiff**2 + np.trace(covDiff)
    return fid


def saveGeneratedAudio(textDataset, checkpointDir):
    featureNet = CBHG(BATCH_SIZE, 16, True, 768)
    initFNetTensor = tf.random.normal((BATCH_SIZE, 768, 1))
    initFNet = featureNet(initFNetTensor)
    featureNet.load_weights(os.path.join(CKPT_DIR, "fnet.keras"))
    embeddings, _ = featureNet(textDataset)
    generator = Generator(BATCH_SIZE, True)
    initGenTensor = tf.random.normal((BATCH_SIZE, 400, 768))
    noise = tf.random.normal((BATCH_SIZE, 128, 1))
    initGen = generator(initGenTensor, noise)
    generator.load_weights(os.path.join(CKPT_DIR, "gen.keras"))
    generatedAudio = generator(embeddings, noise)
    for i in range(len(generatedAudio)):
        audio = tf.reshape(generatedAudio[i], (48000))
        sf.write(os.path.join(GENERATED_WAVS_DIR, "generatedWav"+str(i+1)+".wav"), audio, 24000)

def getAllSamples(wavsDir, generatedWavsDir):
    realAudioList, generatedAudioList = [], []
    for wav, generatedWav in zip(os.listdir(wavsDir), os.listdir(generatedWavsDir)):
        audio, _ = librosa.load(wavsDir + '/' + wav, sr=24000)
        duration = librosa.get_duration(audio, sr=24000)
        offset = np.random.randint(0, duration - 2)
        audio, _ = librosa.load(wavsDir + '/' + wav, sr=24000, offset=int(offset), duration=2)
        quantizedAudio = librosa.mu_compress(audio)
        realAudioList.append(quantizedAudio)
        fakeAudio, _ = librosa.load(generatedWavsDir + '/' + generatedWav, sr=24000)
        generatedAudioList.append(fakeAudio)
    return realAudioList, generatedAudioList



if __name__=='__main__':
    #audioDataset, textDataset = getDataset(WAVS_DIR, TEXT_DIR)
    #saveGeneratedAudio(textDataset, CKPT_DIR)
    realAudio, generatedAudio = getAllSamples(WAVS_DIR, GENERATED_WAVS_DIR)
    print(FrechetVGGDistance(realAudio, generatedAudio))
