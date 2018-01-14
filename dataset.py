from tensorflow.examples.tutorials.mnist import input_data

class Dataset:
	def __init__(self, name, test_size=1000):
		self.name = name
		self.mnist = None
		self.true_transform = None
		self.test_size = test_size
		if self.name == 'mnist':
			data_dir = '/tmp/tensorflow/mnist/input_data'
			self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
			self.test_X = self.mnist.test.images
			self.test_Y = self.mnist.test.labels
		elif self.name.startswith('true'):
			self.true_transform = gen_matrix(n, self.name.split("true_",1)[1] )
			test_X, test_Y = gen_batch(self.true_transform, self.test_size)
			self.test_X = test_X	
			self.test_Y = test_Y
		else:
			print 'Not supported: ', self.name
			assert 0
	def batch(self, batch_size):
		if self.name == 'mnist':
			batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
			return batch_xs, batch_ys
		elif self.name.startswith('true'):
			return gen_batch(self.true_transform, batch_size)
		else:
			print 'Not supported: ', name
			assert 0

