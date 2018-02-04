from utils import *
import datetime, os

class ModelParams:
	def __init__(self, dataset_name, log_path, n, out_size, num_layers, loss, r, steps, batch_size, 
			lr, mom, init_type, class_type, disp_type, learn_corner, 
			n_diag_learned, init_stddev, fix_G):
		if disp_type not in ['stein', 'sylvester']:
			print 'Displacement type ' + disp_type + ' not supported'
			assert 0
		if class_type not in ['toeplitz_like', 'hankel_like', 'vandermonde_like', 'unconstrained', 'circulant_sparsity', 'tridiagonal_corner']:
			print 'Class type ' + class_type + ' not supported'
			assert 0
		self.dataset_name = dataset_name
		self.log_path = log_path
		self.n = n
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
		self.disp_type = disp_type
		self.class_type = class_type
		self.learn_corner = learn_corner
		self.n_diag_learned = n_diag_learned
		self.init_stddev = init_stddev
	
	def save(self, results_dir, name):
		# Append git commit ID
		commit_id = get_commit_id()
		param_str = commit_id + '\n' + str(self)

		# Make new dir with timestamp
		this_results_dir = results_dir + name + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
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
