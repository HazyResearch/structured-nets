from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder
sys.path.insert(0, '../../../../')
from utils import *


class Dataset:
	# true_test: if True, we test on test set. Otherwise, split training set into train/validation.
	def __init__(self, name, n, test_size=1000, true_test=False):
		self.name = name
		self.mnist = None
		self.true_transform = None
		self.test_size = test_size
		self.true_test = true_test
		self.n = n
		if self.name == 'mnist':
			data_dir = '/tmp/tensorflow/mnist/input_data'
			self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
			self.test_X = self.mnist.test.images
			self.test_Y = self.mnist.test.labels
		elif self.name == 'mnist_rot':
			train_data_loc = '/dfs/scratch1/thomasat/datasets/mnist_rot/mnist_all_rotation_normalized_float_train_valid.amat'
			test_data_loc = '/dfs/scratch1/thomasat/datasets/mnist_rot/mnist_all_rotation_normalized_float_test.amat'
			self.load_data(train_data_loc, test_data_loc)
		elif self.name == 'mnist_bg_rot':
			train_data_loc = '/dfs/scratch1/thomasat/datasets/mnist_bg_rot/mnist_all_background_images_rotation_normalized_train_valid.amat'
			test_data_loc = '/dfs/scratch1/thomasat/datasets/mnist_bg_rot/mnist_all_background_images_rotation_normalized_test.amat'
			self.load_data(train_data_loc, test_data_loc)
		elif self.name == 'mnist_rand_bg':
			train_data_loc = '/dfs/scratch1/thomasat/datasets/mnist_rand_bg/mnist_background_random_train.amat'
			test_data_loc = '/dfs/scratch1/thomasat/datasets/mnist_rand_bg/mnist_background_random_test.amat'
			self.load_data(train_data_loc, test_data_loc)
		elif self.name.startswith('mnist_noise'):
			idx = self.name[-1]
			train_data_loc = '/dfs/scratch1/thomasat/datasets/mnist_noise/mnist_noise_variations_all_' + idx + '.amat'
			test_data_loc = '/dfs/scratch1/thomasat/datasets/mnist_noise/mnist_noise_variations_all_' + idx + '.amat'
			self.test_size = 2000 # As specified in http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007#Downloadable_datasets
			self.load_data(train_data_loc, test_data_loc)
		elif self.name == 'convex':
			train_data_loc = '/dfs/scratch1/thomasat/datasets/convex/convex_train.amat'
			test_data_loc = '/dfs/scratch1/thomasat/datasets/convex/50k/convex_test.amat'
			self.load_data(train_data_loc, test_data_loc)
		elif self.name == 'rect':
			train_data_loc = '/dfs/scratch1/thomasat/datasets/rect/rectangles_train.amat'
			test_data_loc = '/dfs/scratch1/thomasat/datasets/rect/rectangles_test.amat'
			self.load_data(train_data_loc, test_data_loc)
		elif self.name == 'rect_images':
			train_data_loc = '/dfs/scratch1/thomasat/datasets/rect_images/rectangles_im_train.amat'
			test_data_loc = '/dfs/scratch1/thomasat/datasets/rect_images/rectangles_im_test.amat'
			self.load_data(train_data_loc, test_data_loc)
		elif self.name.startswith('true'):
			self.true_transform = gen_matrix(n, self.name.split("true_",1)[1] )
			test_X, test_Y = gen_batch(self.true_transform, self.test_size)
			self.test_X = test_X	
			self.test_Y = test_Y
		else:
			print 'Not supported: ', self.name
			assert 0
	
	def out_size(self):
		if 'mnist' in self.name:
			return 10
		elif self.name in ['convex', 'rect', 'rect_images']:
			return 2
		else:
			return self.n

	def load_data(self, train_loc, test_loc):
		if train_loc == test_loc:
			assert not self.true_test or self.name.startswith('mnist_noise')

		if self.true_test and train_loc != test_loc:
			train_data = np.genfromtxt(train_loc)
			test_data = np.genfromtxt(test_loc)

			self.train_X = train_data[:, :-1]
			self.train_Y = np.expand_dims(train_data[:, -1], 1)
			self.test_X = test_data[:, :-1]
			self.test_Y = np.expand_dims(test_data[:, -1], 1)

			# Y must be one-hot
			enc = OneHotEncoder()
			self.train_Y = enc.fit_transform(self.train_Y).todense()
			self.test_Y = enc.fit_transform(self.test_Y).todense()

		else: # Use train_loc only
			data = np.genfromtxt(train_loc)
			# Last self.test_size examples are for validation
			# Last column is true class

			self.train_X = data[:-self.test_size, :-1]
			self.train_Y = np.expand_dims(data[:-self.test_size, -1], 1)
			self.test_X = data[-self.test_size:, :-1]
			self.test_Y = np.expand_dims(data[-self.test_size:, -1], 1)

			# Y must be one-hot
			enc = OneHotEncoder()
			self.train_Y = enc.fit_transform(self.train_Y).todense()
			self.test_Y = enc.fit_transform(self.test_Y).todense()

		print self.train_X.shape, self.train_Y.shape, self.test_X.shape, self.test_Y.shape
		quit()
	def batch(self, batch_size):
		if self.name == 'mnist':
			batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
			return batch_xs, batch_ys
		elif self.name.startswith('mnist') or self.name in ['convex', 'rect', 'rect_images']:
			# Randomly sample batch_size from train_X and train_Y
			idx = np.random.randint(self.train_X.shape[0], size=batch_size)
			return self.train_X[idx, :], self.train_Y[idx, :]
		elif self.name.startswith('true'):
			return gen_batch(self.true_transform, batch_size)
		else:
			print 'Not supported: ', name
			assert 0

