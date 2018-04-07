from utils import *
import datetime, os

class ModelParams:
	def __init__(self, dataset_name, transform, test, log_path, input_size, 
			layer_size, out_size, num_layers, loss, r, steps, batch_size, 
			lr, mom, init_type, class_type, learn_corner, n_diag_learned, 
			init_stddev, fix_G, check_disp, checkpoint_freq, checkpoint_path, 
			test_freq, verbose, decay_rate, decay_freq, learn_diagonal, 
			fix_A_identity, stochastic_train, flip_K_B, num_conv_layers, 
			torch, model):
		if class_type not in ['symmetric', 'polynomial_transform', 'low_rank', 'toeplitz_like', 'hankel_like', 'vandermonde_like', 'unconstrained', 'circulant_sparsity', 'tridiagonal_corner']:
			print('Class type ' + class_type + ' not supported')
			assert 0
		self.dataset_name = dataset_name
		# grayscale
		self.transform = transform
		self.test = test
		self.log_path = log_path
		self.input_size = input_size
		self.layer_size = layer_size
		self.out_size = out_size
		self.num_layers = num_layers
		self.loss = loss
		self.r = r
		self.fix_G = fix_G
		self.steps = steps
		self.batch_size = batch_size
		self.lr = lr
		self.mom = mom
		self.init_type = init_type
		self.disp_type = 'stein'
		if class_type == 'toeplitz_like':
			disp_type = 'sylvester'
		self.class_type = class_type
		self.learn_corner = learn_corner
		self.n_diag_learned = n_diag_learned
		self.init_stddev = init_stddev
		self.check_disp = check_disp
		self.checkpoint_freq = checkpoint_freq
		self.checkpoint_path = checkpoint_path
		self.test_freq = test_freq
		self.verbose = verbose
		self.decay_rate = decay_rate
		self.decay_freq = decay_freq
		self.learn_diagonal = learn_diagonal
		self.fix_A_identity = fix_A_identity
		self.stochastic_train = stochastic_train
		self.flip_K_B = flip_K_B
		self.num_conv_layers = num_conv_layers
		self.torch = torch
		self.model = model
		# c1_filters, c1_ksize, p1_size, p1_strides, c2_filters, c2_ksize, p2_size, p2_strides 
		self.set_cnn_params()


	def set_cnn_params(self):
		cnn_params = {}
		if self.dataset_name.startswith('mnist_noise') or self.dataset_name == 'norb':
			cnn_params['c1_ksize'] = 5
			cnn_params['p1_size'] = 2
			cnn_params['p1_strides'] = 2 
			cnn_params['c2_ksize'] = 5
			cnn_params['p2_size'] = 2
			cnn_params['p2_strides'] = 2
			cnn_params['c1_filters'] = 6
			cnn_params['c2_filters'] = 16
			cnn_params['p2_flat_size'] = 7 * 7 * cnn_params['c2_filters']
			self.cnn_params = cnn_params
			
		elif self.dataset_name == 'cifar10':
			cnn_params['c1_ksize'] = 5
			cnn_params['p1_size'] = 2
			cnn_params['p1_strides'] = 2 
			cnn_params['c2_ksize'] = 5
			cnn_params['p2_size'] = 2
			cnn_params['p2_strides'] = 2
			cnn_params['c1_filters'] = 6
			cnn_params['c2_filters'] = 16
			cnn_params['p2_flat_size'] = 8 * 8 * cnn_params['c2_filters']
			self.cnn_params = cnn_params
		
		elif self.dataset_name.startswith('true'):
			self.cnn_params = cnn_params
		#elif self.dataset_name.startswith('norb'):
		#	cnn_params['c1_filters'] = 9
		#	cnn_params['c2_filters'] = 9
		#	cnn_params['p1_size'] = 3
		#	cnn_params['p1_strides'] = 3
		#	cnn_params['p2_size'] = 1
		#	cnn_params['p2_strides'] = 1
		#	cnn_params['p2_flat_size'] = 9 * 9 * cnn_params['c2_filters']
		else:
			print('dataset_name not supported: ', self.dataset_name)
			assert 0



	def save(self, results_dir, name, commit_id):
		# Append git commit ID
		param_str = ''#commit_id + '\n' + str(self)

		print(param_str)

		# Make new dir with timestamp
		this_results_dir = os.path.join(results_dir, name + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
		if not os.path.exists(this_results_dir):
		    os.makedirs(this_results_dir)

		text_file = open(os.path.join(this_results_dir, self.class_type + '_params.txt'), "w")
		text_file.write(param_str)
		text_file.close()

		return this_results_dir

	def __str__(self):
		attr_dict = self.__dict__
		param_str = ''
		for attr in attr_dict:
			param_str += attr + ': ' + str(attr_dict[attr]) + '\n'
		return param_str
