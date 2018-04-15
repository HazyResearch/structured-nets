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
import multiprocessing

# Available datasets: norb, cifar10, smallnorb, mnist, mnist_noise_variation_*, mnist_rand_bg, mnist_bg_rot, convex, rect, rect_images
# Example command: 
# python compare.py --name=test --methods=tridiagonal_corner,toeplitz-like --dataset=true_toeplitz --result_dir=2_25_18 --r=1 --lr=1e-3 --decay_rate=1.0 --decay_steps=0.1 --mom=0.9 --steps=50000 --batch_size=1024 --test=0 --layer_size=50 --transform=none --torch=1 --model=Attention

method_map = {'circulant_sparsity': 'cs', 'tridiagonal_corner': 'tc', 'low_rank': 'lr', 'unconstrained': 'u', 
	'toeplitz_like': 't', 'hankel_like': 'h', 'vandermonde_like': 'v'}

# Command line params
parser = argparse.ArgumentParser()
parser.add_argument("--name") # Name of run
parser.add_argument("--methods") # Which methods
parser.add_argument("--dataset") # Which dataset
parser.add_argument("--result_dir") # Where to save results
parser.add_argument("--r") # Rank / displacement ranks
parser.add_argument('--lr') # Learning rates
parser.add_argument('--decay_rate') # Decay rates of learning rate
parser.add_argument('--decay_freq', type=float) # Decay steps
parser.add_argument('--mom') # Momentums
parser.add_argument('--steps', type=int) # Steps
parser.add_argument('--batch_size', type=int) # Batch size
parser.add_argument('--test', type=int) # Test on test set
parser.add_argument('--layer_size', type=int) # Size of hidden layer
parser.add_argument('--transform') # Any transform of dataset, e.g. grayscale
parser.add_argument('--torch', type=int) # Pytorch or TF
parser.add_argument('--model') # Which model, e.g. CNN, MLP, RNN
args = parser.parse_args()


methods = args.methods.split(',')
ranks = [int(r) for r in args.r.split(',')]
lrs = [float(lr) for lr in args.lr.split(',')]
decay_rates = [float(dr) for dr in args.decay_rate.split(',')]
moms = [float(mom) for mom in args.mom.split(',')]

print('Testing methods: ', methods)
print('Testing ranks: ', ranks)
print('Testing lrs: ', lrs)
print('Testing decay rates: ', decay_rates)
print('Testing moms: ', moms)

# Fixed params
num_layers = 1
out_dir = '/dfs/scratch1/thomasat/'
loss = 'cross_entropy'
test_size = 1000
train_size = 10000
verbose = False
check_disp = False
fix_G = False
fix_A_identity = False
flip_K_B = False
init_type = 'toeplitz'
init_stddev = 0.01 # Random initializations
test_freq = 100
learn_corner = True
learn_diagonal = False
stochastic_train = False
checkpoint_freq = 1000
num_conv_layers = 2
n_trials = 3
log_path = os.path.join(out_dir, 'tensorboard', args.result_dir)
results_dir = os.path.join(out_dir, 'results', args.result_dir) 
checkpoint_path = os.path.join(out_dir, 'checkpoints', args.result_dir)

dataset = Dataset(args.dataset, args.layer_size, args.steps, args.transform, 
	stochastic_train, test_size, train_size, args.test)
n_diag_learned = dataset.input_size - 1
commit_id = get_commit_id()

for method in methods:
	for rank in ranks:
		for lr in lrs:
			for decay_rate in decay_rates: 
				for mom in moms:
					params = ModelParams(args.dataset, args.transform, args.test, log_path, 
							dataset.input_size, args.layer_size, dataset.out_size(), num_layers, 
							loss, rank, args.steps, args.batch_size, lr, mom, init_type, 
							method, learn_corner, n_diag_learned, init_stddev, fix_G, 
							check_disp, checkpoint_freq, checkpoint_path, test_freq, verbose, 
							decay_rate, args.decay_freq, learn_diagonal, fix_A_identity, 
							stochastic_train, flip_K_B, num_conv_layers, args.torch, args.model)

					# Save params + git commit ID
					this_id = args.name + '_' + method_map[method] + '_r' + str(rank) + '_lr' + str(lr) + '_dr' + str(decay_rate) + '_mom' + str(mom) 
					this_results_dir = params.save(results_dir, this_id, commit_id)
					print('this_results_dir: ', this_results_dir)

					for test_iter in range(n_trials):
						this_iter_name = this_id + '_' + str(test_iter)
						params.log_path = os.path.join(log_path, this_iter_name)
						params.checkpoint_path = os.path.join(checkpoint_path, this_iter_name)

						print('Tensorboard log path: ', params.log_path)
						print('Tensorboard checkpoint path: ', params.checkpoint_path)
						if not os.path.exists(params.checkpoint_path):
							os.makedirs(params.checkpoint_path)

						losses, accuracies = optimize(dataset, params)
						tf.reset_default_graph()

						out_loc = os.path.join(this_results_dir, this_iter_name)
						pkl.dump(losses, open(out_loc + '_losses.p', 'wb'), protocol=2)
						pkl.dump(accuracies, open(out_loc + '_accuracies.p', 'wb'), protocol=2)

						print('Saved losses and accuracies for ' + method + ' to: ' + out_loc)

