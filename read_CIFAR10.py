import numpy as np

def read_CIFAR10(filename):
	import cPickle
	fo = open(filename, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	data_x = dict['data']
	labels = dict['labels']
	batch_length = len(data_x)
	data_y = np.zeros([batch_length,10])
	for i in range(batch_length):
		data_y[i,labels[i]] = 1
	return data_x, data_y

def get_CIFAR_data(is_train=True):
	if is_train is False:
		x_test, y_test = read_CIFAR10('test_batch')
		return x_test, y_test
	else:
		x_train, y_train = read_CIFAR10('data_batch_1')
		for i in range(2,6):
			x1_train, y1_train = read_CIFAR10('data_batch_'+str(i))
			x_train = np.concatenate((x_train, x1_train), axis=0)
			y_train = np.concatenate((y_train, y1_train), axis=0)
		return x_train, y_train ##output shape: X=[# of data, # of pixels], Y=[# of data, 10]
