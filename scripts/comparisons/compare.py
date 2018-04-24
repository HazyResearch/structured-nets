"""
Compare methods and hyperparameter settings sequentially.
"""

import sys, os, datetime
import pickle as pkl
sys.path.insert(0, '../../')
from optimize import optimize
from utils import *
from model_params import ModelParams
from dataset import Dataset
import argparse
import threading
import logging

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')

# Available datasets: norb, cifar10, smallnorb, mnist, mnist_noise_variation_*, mnist_rand_bg, mnist_bg_rot, convex, rect, rect_images
# Example command:
# python compare.py --name=test --methods=tridiagonal_corner,toeplitz_like --dataset=true_toeplitz --result_dir=2_25_18 --r=1 --lr=1e-3 --decay_rate=1.0 --decay_steps=0.1 --mom=0.9 --steps=50000 --batch_size=1024 --test=0 --layer_size=50 --transform=none --torch=1 --model=Attention

method_map = {'circulant_sparsity': 'cs', 'tridiagonal_corner': 'tc', 'tridiagonal_corners': 'tcs', 'low_rank': 'lr', 'unconstrained': 'u',
         'toeplitz_like': 't', 'toep_corner': 't1', 'toep_nocorn': 't0', 'hankel_like': 'h', 'vandermonde_like': 'v'}

def compare(args, method, rank, lr, decay_rate, mom):
    params = ModelParams(args.dataset, args.transform, args.test, log_path, 
            dataset.input_size, args.layer_size, dataset.out_size(), num_layers, 
            loss, rank, args.steps, args.batch_size, lr, mom, init_type, 
            method, learn_corner, n_diag_learned, init_stddev, fix_G, 
            check_disp, check_disp_freq, checkpoint_freq, checkpoint_path, test_freq, verbose, 
            decay_rate, args.decay_freq, learn_diagonal, fix_A_identity, 
            stochastic_train, flip_K_B, num_conv_layers, args.torch, args.model,
            viz_freq, num_pred_plot, viz_powers, early_stop_steps)

    # Save params + git commit ID
    this_id = args.name + '_' + method_map[method] + '_r' + str(rank) + '_lr' + str(lr) + '_dr' + str(decay_rate) + '_mom' + str(mom) + '_bs' + str(args.batch_size)
    this_results_dir = params.save(results_dir, this_id, commit_id)

    for test_iter in range(args.trials):
        this_iter_name = this_id + '_' + str(test_iter)
        params.log_path = os.path.join(log_path, this_iter_name)
        params.checkpoint_path = os.path.join(checkpoint_path, this_iter_name)
        params.vis_path = os.path.join(vis_path, this_iter_name)
        params.result_path = os.path.join(this_results_dir,this_iter_name)

        logging.debug('Tensorboard log path: ' + params.log_path)
        logging.debug('Tensorboard checkpoint path: ' + params.checkpoint_path)
        logging.debug('Tensorboard vis path: ' + params.vis_path)
        logging.debug('Results dir: ' + params.result_path)

        if not os.path.exists(params.checkpoint_path):
            os.makedirs(params.checkpoint_path)


        if not os.path.exists(params.vis_path):
            os.makedirs(params.vis_path)

        losses, accuracies = optimize(dataset, params)
        tf.reset_default_graph()

        pkl.dump(losses, open(params.result_path + '_losses.p', 'wb'), protocol=2)
        pkl.dump(accuracies, open(params.result_path + '_accuracies.p', 'wb'), protocol=2)

        logging.debug('Saved losses and accuracies for ' + method + ' to: ' + params.result_path)

# Command line params
parser = argparse.ArgumentParser()
parser.add_argument("--name") # Name of run
parser.add_argument("--methods") # Which methods
parser.add_argument("--dataset") # Which dataset
parser.add_argument("--result_dir") # Where to save results
parser.add_argument("--r") # Rank / displacement ranks
parser.add_argument('--lr') # Learning rates
parser.add_argument('--decay_rate', default=1.0) # Decay rates of learning rate
parser.add_argument('--decay_freq', type=float) # Decay steps
parser.add_argument('--mom') # Momentums
parser.add_argument('--steps', type=int) # Steps
parser.add_argument('--batch_size', type=int) # Batch size
parser.add_argument('--test', type=int, default=1) # Test on test set
parser.add_argument('--layer_size', type=int) # Size of hidden layer
parser.add_argument('--transform', default='none') # Any transform of dataset, e.g. grayscale
parser.add_argument('--torch', type=int) # Pytorch or TF
parser.add_argument('--model') # Which model, e.g. CNN, MLP, RNN
parser.add_argument('--parallel') #
parser.add_argument('--trials', type=int, default=3) #
args = parser.parse_args()


methods = args.methods.split(',')
ranks = [int(r) for r in args.r.split(',')]
lrs = [float(lr) for lr in args.lr.split(',')]
decay_rates = [float(dr) for dr in args.decay_rate.split(',')]
moms = [float(mom) for mom in args.mom.split(',')]

logging.debug('Testing methods: ' + str(methods))
logging.debug('Testing ranks: ' + str(ranks))
logging.debug('Testing lrs: ' + str(lrs))
logging.debug('Testing decay rates: ' + str(decay_rates))
logging.debug('Testing moms: ' + str(moms))

# Fixed params
num_layers = 1
out_dir = '../..'
loss = 'cross_entropy'
test_size = 1000
train_size = 10000
verbose = False
check_disp = False # If true, checks rank of error matrix every check_disp_freq iters
check_disp_freq = 5000 
fix_G = False
early_stop_steps = 500000
fix_A_identity = False
flip_K_B = False
init_type = 'toeplitz'
init_stddev = 0.01 # Random initializations
test_freq = 100
viz_freq = -1#1000
num_pred_plot = 5
viz_powers = [1,5,10]
learn_corner = True
learn_diagonal = False
stochastic_train = False
checkpoint_freq = 1000
num_conv_layers = 2
# trials = 3
log_path = os.path.join(out_dir, 'tensorboard', args.result_dir)
results_dir = os.path.join(out_dir, 'results', args.result_dir)
checkpoint_path = os.path.join(out_dir, 'checkpoints', args.result_dir)
vis_path = os.path.join(out_dir, 'vis', args.result_dir)

dataset = Dataset(args.dataset, args.layer_size, args.steps, args.transform,
    stochastic_train, test_size, train_size, args.test)
n_diag_learned = dataset.input_size - 1
commit_id = get_commit_id()

# TODO use itertools.product to do this
for method in methods:
    for rank in ranks:
        for lr in lrs:
            for decay_rate in decay_rates:
                for mom in moms:
                    if args.parallel:
                        logging.debug('Starting thread')
                        threading.Thread(target=compare,args=(args, method, rank, lr, decay_rate, mom),).start()
                    else:
                        compare(args, method, rank, lr, decay_rate, mom)
