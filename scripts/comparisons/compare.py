"""
Compare methods.
"""

import sys, os, datetime
import pickle as pkl
sys.path.insert(0, '../../')
from optimize import optimize
from utils import *
from model_params import ModelParams
from dataset import Dataset
import argparse

# Available datasets: mnist, mnist_noise_variation_*, mnist_rand_bg, mnist_bg_rot, convex, rect, rect_images
# Example command: 
# python compare.py test toeplitz_like true_toeplitz 2_25_18 1 1e-3 1.0 0.9 0 50 downsample

# Command line params
parser = argparse.ArgumentParser()
parser.add_argument("name") # Name of run
parser.add_argument("method") # Which method
parser.add_argument("dataset") # Which dataset
parser.add_argument("result_dir") # Where to save results
parser.add_argument("r", type=int) # Rank / displacement rank
parser.add_argument('lr', type=float) # Learning rate
parser.add_argument('decay_rate', type=float) # Decay of learning rate
parser.add_argument('mom', type=float) # Momentum
parser.add_argument('test', type=int) # Test on test set
parser.add_argument('layer_size', type=int) # Size of hidden layer
parser.add_argument('transform') # Any transform of dataset, e.g. grayscale
args = parser.parse_args()

# Fixed params
num_layers = 1
out_dir = '../../results/'#'/dfs/scratch1/thomasat/'
loss = 'mse'
steps = 50000
decay_freq = 0.1
batch_size = 50
test_size = 1000
verbose = False
check_disp = False
fix_G = False
fix_A_identity = True
init_type = 'toeplitz'
init_stddev = 0.01 # Random initializations
test_freq = 100
learn_corner = True
learn_diagonal = False
checkpoint_freq = 100000
n_trials = 3
log_path = os.path.join(out_dir, 'tensorboard', args.result_dir)
results_dir = os.path.join(out_dir, 'results', args.result_dir) 
checkpoint_path = os.path.join(out_dir, 'checkpoints', args.result_dir)

dataset = Dataset(args.dataset, args.layer_size, steps, args.transform, test_size, args.test)
n_diag_learned = dataset.input_size - 1

params = ModelParams(args.dataset, args.transform, args.test, log_path, dataset.input_size, args.layer_size, 
		dataset.out_size(), num_layers, loss, args.r, steps, batch_size, 
		args.lr, args.mom, init_type, args.method, learn_corner, 
		n_diag_learned, init_stddev, fix_G, check_disp, checkpoint_freq, 
		checkpoint_path, test_freq, verbose, args.decay_rate, decay_freq, learn_diagonal, fix_A_identity)

print 'Params:\n', params

# Save params + git commit ID
this_results_dir = params.save(results_dir, args.name)

for test_iter in range(n_trials):
	this_iter_name = args.method + str(test_iter)
	params.log_path = os.path.join(log_path, args.name + '_' + str(test_iter))
	params.checkpoint_path = os.path.join(checkpoint_path, args.name + '_' + str(test_iter))

	print 'Tensorboard log path: ', params.log_path
	print 'Tensorboard checkpoint path: ', params.checkpoint_path

	losses, accuracies = optimize(dataset, params)
	tf.reset_default_graph()

	out_loc = os.path.join(this_results_dir, this_iter_name)
	pkl.dump(losses, open(out_loc + '_losses.p', 'wb'))
	pkl.dump(accuracies, open(out_loc + '_accuracies.p', 'wb'))

	print 'Saved losses and accuracies for ' + args.method + ' to: ' + out_loc

