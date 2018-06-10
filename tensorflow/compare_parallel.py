"""
Compare methods in parallel, spawning separate thread for each.
"""

import sys, os, datetime
import pickle as pkl
sys.path.insert(0, '../../')
# from optimize import optimize
from utils import *
from model_params import ModelParams
from dataset import Dataset
import argparse
import thread

def create_command(args,method,rank,lr,decay_rate,mom):
	command = 'python compare.py --name=%s --methods=%s --dataset=%s --result_dir=%s --r=%s --lr=%s --decay_rate=%s --decay_freq=%s --mom=%s --steps=%s --batch_size=%s --test=%s --layer_size=%s --transform=%s --torch=%s --model=%s'

	return command % (args.name, method, args.dataset, args.result_dir, rank, lr, decay_rate, args.decay_freq, mom, args.steps, args.batch_size, args.test, args.layer_size, args.transform, args.torch, args.model)

# python compare_parallel.py --name=test --methods=tridiagonal_corner,toeplitz-like --dataset=true_toeplitz --result_dir=2_25_18 --r=1 --lr=1e-3 --decay_rate=1.0 --decay_steps=0.1 --mom=0.9 --steps=50000 --batch_size=1024 --test=0 --layer_size=50 --transform=none --torch=1 --model=Attention

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

for method in methods:
	for rank in ranks:
		for lr in lrs:
			for decay_rate in decay_rates: 
				for mom in moms:
					command = create_command(args,method,rank,lr,decay_rate,mom)
					print('Starting new thread:\n', command)
					#os.system(command)
					thread.start_new_thread(os.system, (command,))
