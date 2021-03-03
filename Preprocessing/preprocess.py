import os
import pandas as pd
import numpy as np
import shutil
import librosa


RAW_DATASET_DIR = 'LJSpeech-1.1'
CLEAN_DATASET_DIR = 'LJSpeech'
CSV_NAME = 'text.csv'
TEXT_DIR = 'texts'
WAVS_DIR = 'wavs'


def copyAndFilterWavs(rawDatasetDir, cleanDatasetDir):
        wavPath = rawDatasetDir + '/wavs'
        for file in os.listdir(wavPath):
            audio, _ = librosa.load(os.path.join(wavPath, file), sr=24000)
            duration = int(librosa.get_duration(audio, sr=24000))
            if duration < 3:
                os.remove(os.path.join(cleanDatasetDir + '/texts', file.split('.')[0] + '.txt'))
            else:
                shutil.copy(os.path.join(wavPath, file), os.path.join(cleanDatasetDir+'/wavs', file))
        print("Copied all files")

def makeCSV(rawDatasetDir, csvName):
    for file in os.listdir(rawDatasetDir):
        if file.endswith(".csv"):
            with open(os.path.join(rawDatasetDir, file), 'r+', encoding='utf-8') as metadata:
                s = "Id|Text|Normalized text"
                data = metadata.read()
                myCSV = open(csvName, "w+", encoding='utf-8')
                myCSV.write(s.strip('\r\n') + '\n')
                myCSV.write(data)
    print("Generated", csvName)


def makeTextWavDirectory(cleanDatasetDir, wavDir, textDir):
    if not os.path.exists(cleanDatasetDir):
        textPath = os.path.join(cleanDatasetDir, textDir)
        wavPath = os.path.join(cleanDatasetDir, WAVS_DIR)
        os.makedirs(textPath)
        os.makedirs(wavPath)
    else:
        print("Clean data set already exists")

def copyTexts(cleanDatasetDir, csvName, textDir):
    textPath = os.path.join(cleanDatasetDir, textDir)
    df = pd.read_csv(csvName, sep='|')
    df = df.drop(columns=['Text'])
    expansion = {
        "Mr.": "Mister",
        "Mrs.": "Misess",
        "Dr.": "Doctor",
        "No.": "Number",
        "St.": "Saint",
        "Co.": "Company",
        "Jr.": "Junior",
        "Maj.": "Major",
        "Gen.": "General",
        "Drs.": "Doctors",
        "Rev.": "Reverend",
        "Lt.": "Lieutenant",
        "Hon.": "Honorable",
        "Sgt.": "Sergeant",
        "Capt.": "Captain",
        "Esq.": "Esquire",
        "Ltd.": "Limited",
        "Col.": "Colonel",
        "Ft.": "Fort"
    }
    textExpanded = []
    for text in df['Normalized text']:
        for key in expansion.keys():
            text = str(text).replace(key, expansion[key])
        textExpanded.append(text)
    df['Normalized text'] = textExpanded
    for textId, text in zip(df['Id'], df['Normalized text']):
        f = open(os.path.join(textPath, textId + ".txt"), "w+")
        f.write(text)
        f.close()
        print("Created", textId, "file")
    os.remove(csvName)
    print(csvName, "removed")


if __name__ == '__main__':
    makeCSV(RAW_DATASET_DIR, CSV_NAME)
    makeTextWavDirectory(CLEAN_DATASET_DIR, WAVS_DIR, TEXT_DIR)
    copyTexts(CLEAN_DATASET_DIR, CSV_NAME, TEXT_DIR)
    copyAndFilterWavs(RAW_DATASET_DIR, CLEAN_DATASET_DIR)