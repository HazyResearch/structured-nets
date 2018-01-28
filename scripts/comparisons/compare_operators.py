"""
Compare learned and fixed operators.
"""
import sys, os, datetime
import pickle as pkl
sys.path.insert(0, '../../')
from learned_operators import *
from fixed_operators import *
from utils import *
from model_params import ModelParams
from dataset import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()

n = 784
num_layers = 1
loss = 'cross_entropy'
# Available datasets: mnist, mnist_noise_variation_*, mnist_rand_bg, mnist_bg_rot, convex, rect, rect_images
dataset_name = 'mnist_rand_bg'
steps = 50000
batch_size = 50
test_size = 1000
momentums = [0.9]
learn_rate = 0.002
displacement_rank = 1
learn_corner = True
fix_G = False
n_diag_learneds = [0]
init_stddev = 0.01
init_type = 'toeplitz'
init_stddev = 0.1 # For random initialization
test_freq = 100
n_trials = 5
results_dir = '/dfs/scratch1/thomasat/results/1_27_18/' #'../../results/'#

#Available test_fns: [toeplitz_like, hankel_like, vandermonde_like, unconstrained, circulant_sparsity]
test_fns = [toeplitz_like]#[circulant_sparsity]  
dataset = Dataset(dataset_name, n, test_size)
out_size = dataset.out_size() # 10 for MNIST, 2 for convex, rect, rect_images

# Iterate over 
for mom in momentums:
	for n_diag_learned in n_diag_learneds:
		for fn in test_fns:

			# Current Toeplitz-like is a special case: inversion assumes Sylvester type displacement
			disp_type = 'stein'
			if fn.__name__ == 'toeplitz_like':
				disp_type = 'sylvester'

			params = ModelParams(dataset_name, n, out_size, num_layers, loss, displacement_rank, steps, batch_size, 
					learn_rate, mom, init_type, fn.__name__, disp_type, learn_corner, n_diag_learned, 
					init_stddev, fix_G)
			
			# Save params + git commit ID
			this_results_dir = params.save(results_dir, args.name)

			for test_iter in range(n_trials):
				losses, accuracies = fn(dataset, params, test_freq)

				out_loc = os.path.join(this_results_dir, fn.__name__ + str(test_iter))
				pkl.dump(losses, open(out_loc + '_losses.p', 'wb'))
				pkl.dump(accuracies, open(out_loc + '_accuracies.p', 'wb'))

				print 'Saved losses and accuracies for ' + fn.__name__ + ' to: ' + out_loc

