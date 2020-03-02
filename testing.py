import os
import numpy as np
import math
import librosa
import tflearn
import json

os.system('clear')

# READING SOME PARAM
fo = open('./params.json', 'r')
param = json.load(fo)
fo.close()
yLength = param['vector_length']

PATH_TESTS = './dataset/sudais/test/'
DATA_TESTS = [x for x in os.listdir(PATH_TESTS) if x.endswith('.wav')]

def VektorReshape(param):    
    if len(param) < yLength:
        _ceil = math.ceil((yLength - len(param)) / 2)
        _floor = math.floor((yLength - len(param)) / 2)
        return np.pad(param, (_floor, _ceil), mode='constant', constant_values=0)
    
    if len(param) > yLength:
        _ceil = math.ceil((len(param) - yLength) / 2) * -1
        _floor = math.floor((len(param) - yLength) / 2)
        return param[_floor:_ceil]
    
    return param

# TESTING PART
net = tflearn.input_data([None, param['mfcc_0'], param['mfcc_1']])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 8, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
os.system('clear')
for x in DATA_TESTS:
    y, sr = librosa.load(PATH_TESTS + x, sr=None, mono=True)
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    emphasized_signal = VektorReshape(emphasized_signal)
    mfcc = librosa.feature.mfcc(emphasized_signal, sr)
    model.load(param['model_name'])
    try:
        
        result 	= model.predict([mfcc])
        result 	= list(result[0])
        idx_v 	= max(result)
        idx		= result.index(idx_v)
        print(x)
        print('%d : %.10f' % (idx, idx_v))
    except:
        print('Error!')