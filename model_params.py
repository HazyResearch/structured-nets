from utils import *
import datetime, os

class ModelParams:
	def __init__(self, dataset_name, transform, test, log_path, input_size, 
			layer_size, out_size, num_layers, loss, r, steps, batch_size, 
			lr, mom, init_type, class_type, learn_corner, n_diag_learned, 
			init_stddev, fix_G, check_disp, checkpoint_freq, checkpoint_path, 
			test_freq, verbose, decay_rate, decay_freq, learn_diagonal, 
			fix_A_identity, stochastic_train, flip_K_B):
		if class_type not in ['symmetric', 'polynomial_transform', 'low_rank', 'toeplitz_like', 'hankel_like', 'vandermonde_like', 'unconstrained', 'circulant_sparsity', 'tridiagonal_corner']:
			print 'Class type ' + class_type + ' not supported'
			assert 0
		self.dataset_name = dataset_name
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


	def save(self, results_dir, name):
		# Append git commit ID
		commit_id = get_commit_id()
		param_str = commit_id + '\n' + str(self)

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
