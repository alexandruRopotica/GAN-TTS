import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf


PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
ENCODER = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1"
DISC_LEARNING_RATE = 1e-4
GEN_LEARNING_RATE = 5e-5
BETA_1 = 0
BETA_2 = 0.999
DECAY_RATE = 0.9999
WINDOWS = [240, 480, 960, 1920, 3600]
WINDOWS_TEST = [960, 1920, 3600]
BATCH_SIZE = 1
EPOCHS = 1


def getSamples(audioArray, windows):
    totalSamples = len(audioArray[0])
    subSamples = []
    for window in windows:
        idx = np.random.randint(0, totalSamples - window)
        subSamples.append(audioArray[:, idx:idx+window, :])
    return subSamples


def initializeModels(batchSize):
    bert = BERT(PREPROCESSOR, ENCODER)
    featureNet = CBHG(batchSize, 16, True)
    generator = Generator(batchSize, True)
    discriminatorTest = DiscriminatorTest()
    #discriminator = Discriminator()
    genOptimizer = tfa.optimizers.MovingAverage(decay=DECAY_RATE,
        optimizer=tf.keras.optimizers.Adam(lr=GEN_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2))
    discOptimizer = tf.keras.optimizers.Adam(lr=DISC_LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
    return bert, featureNet, generator, discriminatorTest, genOptimizer, discOptimizer


def trainStep(audioBatch, embeddings, featureNet, generator, discriminator, genOptimizer, discOptimizer):
    with tf.device('/device:GPU:0'):
        noise = tf.random.normal((BATCH_SIZE, 128, 1))
        with tf.GradientTape() as genTape, tf.GradientTape() as discTape, tf.GradientTape() as featureTape:
            genFeatures, discFeatures = featureNet(embeddings)
            generatedAudio = generator(genFeatures, noise)
            # w1, w2, w3, w4, w5 = getSamples(generatedAudio, WINDOWS)
            w3, w4, w5 = getSamples(generatedAudio, WINDOWS_TEST)
            fakeAudio = discriminator(w3, w4, w5, discFeatures)
            # w1, w2, w3, w4, w5 = getSamples(audioBatch, WINDOWS)
            w3, w4, w5 = getSamples(audioBatch, WINDOWS_TEST)
            realAudio = discriminator(w3, w4, w5, discFeatures)
            discFakeLoss = tf.losses.hinge(tf.zeros_like(fakeAudio), fakeAudio)
            discRealLoss = tf.losses.hinge(tf.zeros_like(realAudio), realAudio)
            discLoss = discFakeLoss + discRealLoss
            discGradients = discTape.gradient(discLoss, discriminator.trainable_variables)
            discOptimizer.apply_gradients(zip(discGradients, discriminator.trainable_variables))
            genLoss = tf.losses.hinge(tf.ones_like(fakeAudio), fakeAudio)
            genGradients = genTape.gradient(genLoss, generator.trainable_variables)
            genOptimizer.apply_gradients(zip(genGradients, generator.trainable_variables))
            featureGradients = featureTape.gradient(discLoss, featureNet.trainable_variables)
            discOptimizer.apply_gradients(zip(featureGradients, featureNet.trainable_variables))
            print("Generator loss:", genLoss.numpy(),"| Discriminator loss:", discLoss.numpy())


def train(audioDataset, textDataset, epochs):
    bert, featureNet, generator, discriminator, genOptimizer, discOptimizer = initializeModels(BATCH_SIZE)
    for epoch in range(epochs):
        print("Epoch", epoch+1)
        for audioBatch, textBatch in zip(audioDataset, textDataset):
            audioBatch = tf.expand_dims(audioBatch, axis=0)
            embeddings = bert(textBatch)
            trainStep(audioBatch, embeddings, featureNet, generator, discriminator, genOptimizer, discOptimizer)


if __name__ == '__main__':
    pass