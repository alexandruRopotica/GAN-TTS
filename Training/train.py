import numpy as np


def getSamples(audioArray, windows):
    totalSamples = len(audioArray[0])
    subSamples = []
    for window in windows:
        idx = np.random.randint(0, totalSamples - window)
        subSamples.append(audioArray[:, idx:idx+window, :])
    return subSamples