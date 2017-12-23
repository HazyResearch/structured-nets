"""
Compare learned and fixed operators.
"""
import sys, os
sys.path.insert(0, '../../')
from learn_operators import *
from fixed_operators import *
from utils import *
import pickle as pkl

np.random.seed(0)

n = 50 # True matrix is n x n 
true_transform = 'toeplitz' # Must be 'toeplitz', 'hankel', 'vandermonde', 'cauchy', or 'random'
M = gen_matrix(n, true_transform)
steps = 20000
batch_size = 50
test_size = 1000
momentum = 0.99
learn_rate = 1e-5 
displacement_rank = 2 
results_dir = '../../results/'
fullprefix = results_dir + 'mom' + str(momentum) + '_' + true_transform
test_X, test_Y = gen_batch(M, test_size)

# TODO: create separate results directory for each experiment, with timestamp
# TODO: log settings + git commit ID

# Various settings to test
test_fns = [unconstrained, tridiagonal_corner, circulant_sparsity, toeplitz_like,
	vandermonde_like, hankel_like]

# Create results dir
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for fn in test_fns:
	out_loc = fullprefix + '_' + fn.__name__ + '_losses_' +  str(n) +'.p'
	losses = fn(M, test_X, test_Y, displacement_rank, steps, batch_size, learn_rate, momentum)
	pkl.dump(losses, open(out_loc, 'wb'))