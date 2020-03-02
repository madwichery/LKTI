import os
import numpy as np
import math
import librosa
import tflearn
import json

os.system('clear')

yLength = 0
MFCCS = list()
LABELS = list()
_MFCCS = list()
_LABELS = list()
TRAIN_ITERS = 100
MODEL_NAME = 'tflearn.lstm_with_padding.model'

PATH_TRAIN = './dataset/sudais/train/'
DATA_TRAIN = [x for x in os.listdir(PATH_TRAIN) if x.endswith('.wav')]
PATH_VALID = './dataset/sudais/valid/'
DATA_VALID = [x for x in os.listdir(PATH_VALID) if x.endswith('.wav')]
PATH_TESTS = './dataset/sudais/test/'
DATA_TESTS = [x for x in os.listdir(PATH_TESTS) if x.endswith('.wav')]

for x in DATA_TRAIN:
    y, sr = librosa.load(PATH_TRAIN + x, sr=None, mono=True)
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    if len(emphasized_signal) > yLength:
        yLength = len(emphasized_signal)
        
for x in DATA_VALID:
    y, sr = librosa.load(PATH_VALID + x, sr=None, mono=True)
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    if len(emphasized_signal) > yLength:
        yLength = len(emphasized_signal)
        
for x in DATA_TESTS:
    y, sr = librosa.load(PATH_TESTS + x, sr=None, mono=True)
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    if len(emphasized_signal) > yLength:
        yLength = len(emphasized_signal)
        
print('Length of vectorized data is %s' % yLength)

def VektorReshape(param):    
    if len(param) < yLength:
        _ceil = math.ceil((yLength - len(param)) / 2)
        _floor = math.floor((yLength - len(param)) / 2)
        return np.pad(param, (_floor, _ceil), mode='constant', constant_values=0)
    
    return param

def LABEL_AL_FATIHA(filename):
    ayah = int(filename[5])
    
    return np.eye(8)[ayah]

for x in DATA_TRAIN:
    y, sr = librosa.load(PATH_TRAIN + x, sr=None, mono=True)
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    emphasized_signal = VektorReshape(emphasized_signal)
    mfcc = librosa.feature.mfcc(emphasized_signal, sr)
    LABELS.append(LABEL_AL_FATIHA(x))
    MFCCS.append(np.array(mfcc))
        
for x in DATA_VALID:
    y, sr = librosa.load(PATH_VALID + x, sr=None, mono=True)
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    emphasized_signal = VektorReshape(emphasized_signal)
    mfcc = librosa.feature.mfcc(emphasized_signal, sr)
    _LABELS.append(LABEL_AL_FATIHA(x))
    _MFCCS.append(np.array(mfcc))

os.system('clear')

batchs = list()
for i in range(1, len(MFCCS)+1):
    if len(MFCCS) % i == 0:
        batchs.append(i)
print(batchs)
BATCH = int(input('Please choose a batch size: '))
if BATCH not in batchs:
    print('Choosen batch size: %d' % len(MFCCS))

# LEARNING PART
net = tflearn.input_data([None, MFCCS[0].shape[0], MFCCS[0].shape[1]])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 8, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
os.system('clear')
TRAIN_ITERS = int(input('Input iters: '))
for i in range(TRAIN_ITERS):
    print('ITER NUMBER : %d' % i)
    model.fit(MFCCS, LABELS, n_epoch=10, validation_set=(_MFCCS, _LABELS), show_metric=True, batch_size=BATCH)
MODEL_NAME = './models/tflearn.sudais_lstm_with_padding.model'
model.save(MODEL_NAME)

# WRITING SOME PARAMS
param = dict(
    vector_length = yLength,
    mfcc_0 = MFCCS[0].shape[0],
    mfcc_1 = MFCCS[0].shape[1],
    model_name = MODEL_NAME
)

fo = open('./params.json', 'w')
json.dump(param, fo)
fo.close()