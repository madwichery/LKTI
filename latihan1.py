import os, numpy as np, librosa, tflearn

os.system('clear')

def LABEL_AL_FATIHA(filename):
	ayah = int(filename[5])
	return np.eye(8)[ayah]

PATH_TRAIN = './dataset/train/'
DATA_TRAIN = [x for x in os.listdir(PATH_TRAIN) if x.endswith('.wav')]
PATH_VALID = './dataset/valid/'
DATA_VALID = [x for x in os.listdir(PATH_VALID) if x.endswith('.wav')]
PATH_TESTS = './dataset/test/'
DATA_TESTS = [x for x in os.listdir(PATH_TESTS) if x.endswith('.wav')]
MAX_PAD = 2507
MFCCS = list()
LABELS = list()
_MFCCS = list()
_LABELS = list()
__MFCCS = list()
__LABELS = list()
TRAIN_ITERS = 100

print('Loading training data(s).')
for x in DATA_TRAIN:
	y, sr = librosa.load(PATH_TRAIN + x, sr=None, mono=True)
	emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
	mfcc = librosa.feature.mfcc(emphasized_signal, sr)
	mfcc = np.pad(mfcc, ((0,0), (0,MAX_PAD-len(mfcc[0]))), mode='constant', constant_values=0)
	LABELS.append(LABEL_AL_FATIHA(x))
	MFCCS.append(np.array(mfcc))
	print(x + ' done.')
	#if MAX_PAD < mfcc.shape[1]:
	#    MAX_PAD = mfcc.shape[1]

print('Loading validation data(s).')
for x in DATA_VALID:
	y, sr = librosa.load(PATH_VALID + x, sr=None, mono=True)
	emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
	mfcc = librosa.feature.mfcc(emphasized_signal, sr)
	mfcc = np.pad(mfcc, ((0,0), (0,MAX_PAD-len(mfcc[0]))), mode='constant', constant_values=0)
	_LABELS.append(LABEL_AL_FATIHA(x))
	_MFCCS.append(np.array(mfcc))
	print(x + ' done.')
	#if MAX_PAD < mfcc.shape[1]:
	#    MAX_PAD = mfcc.shape[1]

'''
print('Loading test data(s).')
for x in DATA_TESTS:
	y, sr = librosa.load(PATH_TESTS + x, sr=None, mono=True)
	emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
	mfcc = librosa.feature.mfcc(emphasized_signal, sr)
	mfcc = np.pad(mfcc, ((0,0), (0,MAX_PAD-len(mfcc[0]))), mode='constant', constant_values=0)
	__LABELS.append(LABEL_AL_FATIHA(x))
	__MFCCS.append(np.array(mfcc))
	print(x + ' done.')
	#if MAX_PAD < mfcc.shape[1]:
	#    MAX_PAD = mfcc.shape[1]
'''

#print(MAX_PAD)

batchs = list()
for i in range(1, len(MFCCS)+1):
	if len(MFCCS) % i == 0:
		batchs.append(i)
print(batchs)
BATCH = int(input("Please choose a batch size: "))
if BATCH not in batchs:
	print("Choosen batch size: %d" % len(MFCCS))

net = tflearn.input_data([None, 20, MAX_PAD])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 8, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
for i in range(TRAIN_ITERS):
	model.fit(MFCCS, LABELS, n_epoch=10, validation_set=(_MFCCS, _LABELS), show_metric=True, batch_size=BATCH)
model.save("tflearn.lstm1.model")