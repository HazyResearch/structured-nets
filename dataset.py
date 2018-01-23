from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder
sys.path.insert(0, '../../../../')
from utils import *

class Dataset:
	def __init__(self, name, n, test_size=1000):
		self.name = name
		self.mnist = None
		self.true_transform = None
		self.test_size = test_size
		if self.name == 'mnist':
			data_dir = '/tmp/tensorflow/mnist/input_data'
			self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
			self.test_X = self.mnist.test.images
			self.test_Y = self.mnist.test.labels
		elif self.name == 'mnist_bg_rot':
			data_loc = '/Users/Anna/mnist_rotation_back_image_new/mnist_all_background_images_rotation_normalized_train_valid.amat'
			self.load_mnist_bg_rot(data_loc)
		elif self.name.startswith('true'):
			self.true_transform = gen_matrix(n, self.name.split("true_",1)[1] )
			test_X, test_Y = gen_batch(self.true_transform, self.test_size)
			self.test_X = test_X	
			self.test_Y = test_Y
		else:
			print 'Not supported: ', self.name
			assert 0
	
	def load_mnist_bg_rot(self, data_loc):
		data = np.genfromtxt(data_loc)

		# Last 2k examples are for validation
		# Last column is true class

		self.train_X = data[:-self.test_size, :-1]
		self.train_Y = np.expand_dims(data[:-self.test_size, -1], 1)
		self.test_X = data[-self.test_size:, :-1]
		self.test_Y = np.expand_dims(data[-self.test_size:, -1], 1)

		# Y must be one-hot
		enc = OneHotEncoder()
		self.train_Y = enc.fit_transform(self.train_Y).todense()
		self.test_Y = enc.fit_transform(self.test_Y).todense()

	def batch(self, batch_size):
		if self.name == 'mnist':
			batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
			return batch_xs, batch_ys
		elif self.name == 'mnist_bg_rot':
			# Randomly sample batch_size from train_X and train_Y
			idx = np.random.randint(self.train_X.shape[0], size=batch_size)
			return self.train_X[idx, :], self.train_Y[idx, :]
		elif self.name.startswith('true'):
			return gen_batch(self.true_transform, batch_size)
		else:
			print 'Not supported: ', name
			assert 0

