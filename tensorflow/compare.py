"""
Compare methods and hyperparameter settings sequentially.
"""

import sys, os, datetime
import pickle as pkl
# sys.path.insert(0, '../')
import argparse
import threading
import logging
import numpy as np

from optimize_tf import optimize_tf
from utils import *
from model_params import ModelParams
from dataset import Dataset

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')

# Available datasets: norb, cifar10, smallnorb, mnist, mnist_noise_variation_*, mnist_rand_bg, mnist_bg_rot, convex, rect, rect_images
# Example command:
# python compare.py --name=test --methods=tridiagonal_corner,toeplitz_like --dataset=true_toeplitz --result_dir=2_25_18 --r=1 --lr=1e-3 --decay_rate=1.0 --decay_steps=0.1 --mom=0.9 --steps=50000 --batch_size=1024 --test=0 --layer_size=50 --transform=none --torch=1 --model=Attention

method_map = {'circulant_sparsity': 'cs', 'tridiagonal_corner': 'tc', 'tridiagonal_corners': 'tcs', 'low_rank': 'lr', 'unconstrained': 'u',
              'toeplitz_like': 't', 'toep_corner': 't1', 'toep_nocorn': 't0', 'subdiagonal': 'sd', 'hankel_like': 'h', 'vandermonde_like': 'v'}

def compare(args, method, rank, lr, decay_rate, mom, train_frac, steps):
    params = ModelParams(args.dataset, args.transform, args.test, log_path,
            dataset.input_size, args.layer_size, dataset.out_size(), num_layers,
            loss, rank, steps, args.batch_size, lr, mom, init_type,
            method, learn_corner, n_diag_learned, init_stddev, fix_G,
            check_disp, check_disp_freq, checkpoint_freq, checkpoint_path, test_freq, verbose,
            decay_rate, args.decay_freq, learn_diagonal, fix_A_identity,
            stochastic_train, flip_K_B, num_conv_layers, args.torch, args.model,
            viz_freq, num_pred_plot, viz_powers, early_stop_steps, replacement,
            test_best_val_checkpoint, args.restore, num_structured_layers,
            tie_operators_same_layer, tie_layers_A_A, tie_layers_A_B, train_frac)

    # Save params + git commit ID
    this_id = args.name + '_' + method_map[method] + '_r' + str(rank) + '_lr' + str(lr) + '_dr' + str(decay_rate) + '_mom' + str(mom) + '_bs' + str(args.batch_size) + '_tf' + str(train_frac) + '_steps' + str(steps)
    this_results_dir = params.save(results_dir, this_id, commit_id, command)

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

        losses, accuracies = optimize_tf(dataset, params)
        tf.reset_default_graph()

        pkl.dump(losses, open(params.result_path + '_losses.p', 'wb'), protocol=2)
        pkl.dump(accuracies, open(params.result_path + '_accuracies.p', 'wb'), protocol=2)

        logging.debug('Saved losses and accuracies for ' + method + ' to: ' + params.result_path)


# Command line params
parser = argparse.ArgumentParser()
parser.add_argument("--name", default='') # Name of run
parser.add_argument("--methods") # Which methods
parser.add_argument("--dataset") # Which dataset
parser.add_argument("--result_dir") # Where to save results
parser.add_argument("--r", default='0') # Rank / displacement ranks
parser.add_argument('--lr') # Learning rates
parser.add_argument('--decay_rate', default='1.0') # Decay rates of learning rate
parser.add_argument('--decay_freq', type=float, default=1.0) # Decay steps
parser.add_argument('--mom', default='0.9') # Momentums
parser.add_argument('--steps', type=int) # Steps
parser.add_argument('--batch_size', type=int) # Batch size
parser.add_argument('--test', type=int, default=1) # Test on test set
parser.add_argument('--layer_size', type=int) # Size of hidden layer
parser.add_argument('--transform', default='none') # Any transform of dataset, e.g. grayscale
parser.add_argument('--torch', type=int, default=0) # Pytorch or TF
parser.add_argument('--model') # Which model, e.g. CNN, MLP, RNN
parser.add_argument('--parallel') #
parser.add_argument('--train_frac', default='1.0')
parser.add_argument('--trials', type=int, default=3) #
parser.add_argument('--restore', type=int, default=0) # Whether to restore from latest checkpoint
args = parser.parse_args()


methods = args.methods.split(',')
ranks = [int(r) for r in args.r.split(',')]
lrs = [float(lr) for lr in args.lr.split(',')]
decay_rates = [float(dr) for dr in args.decay_rate.split(',')]
moms = [float(mom) for mom in args.mom.split(',')]
train_fracs = [float(train_frac) for train_frac in args.train_frac.split(',')]

logging.debug('Testing methods: ' + str(methods))
logging.debug('Testing ranks: ' + str(ranks))
logging.debug('Testing lrs: ' + str(lrs))
logging.debug('Testing decay rates: ' + str(decay_rates))
logging.debug('Testing moms: ' + str(moms))
logging.debug('Testing train fracs: ' + str(train_fracs))

# Fixed params
num_layers = 1
out_dir = '../..'
loss = 'cross_entropy'
test_size = 1000
train_size = 10000
verbose = False
replacement = False # If true, sample with replacement when batching
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
test_best_val_checkpoint = True # If true, tests best checkpoint (by validation accuracy). Otherwise, tests last one.
# trials = 3
# Only affect VAE
num_structured_layers = 2
tie_operators_same_layer = False
tie_layers_A_A = False
tie_layers_A_B = False


log_path = os.path.join(out_dir, 'tensorboard', args.result_dir)
results_dir = os.path.join(out_dir, 'results', args.result_dir)
checkpoint_path = os.path.join(out_dir, 'checkpoints', args.result_dir)
vis_path = os.path.join(out_dir, 'vis', args.result_dir)

commit_id = get_commit_id()
command = ' '.join(sys.argv)

# setattr(cf, 'use_cupy', True)

# TODO use itertools.product to do this
for train_frac in train_fracs:
    # Scale steps by train_frac
    this_steps = int(train_frac*args.steps)
    dataset = Dataset(args.dataset, args.layer_size, this_steps, args.transform,
    stochastic_train, replacement, test_size, train_size, args.test, train_frac)
    n_diag_learned = dataset.input_size - 1
    for method in methods:
        for rank in ranks:
            for lr in lrs:
                for decay_rate in decay_rates:
                    for mom in moms:
                        if args.parallel:
                            logging.debug('Starting thread')
                            threading.Thread(target=compare,args=(args, method, rank, lr, decay_rate, mom, train_frac, this_steps),).start()
                        else:
                            compare(args, method, rank, lr, decay_rate, mom, train_frac,this_steps)
