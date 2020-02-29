import os, numpy as np, librosa, tflearn

def LABEL_AL_FATIHA(filename):
	ayah = int(filename[5])
	return np.eye(8)[ayah]

PATH_TEST = './dataset/train/'
DATA_TEST = [x for x in os.listdir(PATH_TEST) if x.endswith('.wav')]
MAX_PAD = 2507

net = tflearn.input_data([None, 20, MAX_PAD])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 8, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
for x in DATA_TEST:
	try:
		y, sr = librosa.load(PATH_TEST + x, sr=None, mono=True)
		emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
		mfcc = librosa.feature.mfcc(emphasized_signal, sr)
		mfcc = np.pad(mfcc, ((0,0), (0,MAX_PAD-len(mfcc[0]))), mode='constant', constant_values=0)
		model.load('tflearn.lstm1.model')
		result 	= model.predict([mfcc])
		result 	= list(result[0])
		idx_v 	= max(result)
		idx		= result.index(idx_v)
		print(x)
		print('%d : %.10f' % (idx, idx_v))
	except:
		print('Error!')