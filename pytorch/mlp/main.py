import sys, os, datetime, subprocess
import pickle as pkl
import itertools
import argparse, argh
import threading
import logging
import pprint
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from inspect import signature

 # add parent (pytorch root) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_params import ModelParams
from dataset import DatasetLoaders
from nets import ArghModel, construct_model
from optimize_torch import optimize_torch
from utils import descendants


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%FT%T',)



# Command line params
parser = argparse.ArgumentParser()
parser.add_argument("--name", default='') # Name of run
parser.add_argument("--dataset") # Which dataset
parser.add_argument('--transform', default='none') # Any transform of dataset, e.g. grayscale
parser.add_argument('--train_frac', nargs='+', default=[None])
parser.add_argument("--result_dir") # Where to save results
# parser.add_argument('--restore', type=int, default=0) # Whether to restore from latest checkpoint
parser.add_argument('--trials', type=int, default=1) #
parser.add_argument('--batch_size', type=int, default=50) # Batch size
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument('--lr', nargs='+', type=float, default=[1e-3]) # Learning rates
parser.add_argument('--mom', nargs='+', type=float, default=[0.9]) # Momentums
parser.add_argument('--val_frac', type=float, default=0.15)
parser.add_argument('--weight_decay', type=float, default=0.975)
parser.add_argument('--log_freq', type=float, default=100)
# parser.add_argument('--steps', type=int) # Steps
parser.add_argument('--test', action='store_false') # Test on test set

out_dir = '../..'

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)



def save_args(args, results_dir):
    commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    command = ' '.join(sys.argv)
    param_str = str(commit_id) + '\n' + command + '\n' + pprint.pformat(vars(args))
    print(param_str)

    # Make new dir with timestamp
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the parameters in readable form
    text_file = open(os.path.join(results_dir, 'params.txt'), "w")
    text_file.write(param_str)
    text_file.close()

    # Save the Namespace object
    pkl.dump(args, open(os.path.join(results_dir, 'params.p'), "wb"))


def mlp(args):
    for train_frac in args.train_frac:
        dataset = DatasetLoaders(args.dataset, args.transform, train_frac, args.val_frac, args.batch_size)
        model = construct_model(nets[args.model], dataset.in_size, dataset.out_size, args)

        for lr, mom in itertools.product(args.lr, args.mom):
            run_name = args.name + '_' + model.name() \
                    + '_lr' + str(lr) \
                    + '_mom' + str(mom) \
                    + '_bs' + str(args.batch_size) \
                    + '_ep' + str(args.epochs) \
                    + '_' + str(args.dataset)
                    # + '_steps' + str(steps)
            if train_frac is not None:
                run_name += '_tf' + str(train_frac)

            results_dir = os.path.join(out_dir,
                                        'results',
                                        args.result_dir,
                                        run_name + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
            save_args(args, results_dir)

            for trial_iter in range(args.trials):
                log_path = os.path.join(out_dir, 'tensorboard', args.result_dir, run_name, str(trial_iter))
                checkpoint_path = os.path.join(out_dir, 'checkpoints', args.result_dir, run_name, str(trial_iter))
                result_path = os.path.join(results_dir, str(trial_iter))
                # vis_path = os.path.join(out_dir, 'vis', args.result_dir, run_iter_name)

                model.reset_parameters()
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
                lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.weight_decay)
                losses, accuracies = optimize_torch(dataset, model, optimizer, lr_scheduler, args.epochs, args.log_freq, log_path, checkpoint_path, result_path, args.test)


## parse

# task
subparsers = parser.add_subparsers()
mlp_parser = subparsers.add_parser('MLP')
mlp_parser.set_defaults(task=mlp)
# vae_parser = subparsers.add_parser('VAE')
# vae_parser.set_defaults(task=vae)
# possible other main commands: sample() for the sample complexity case, vae(), etc.

# MLP models
model_options = []
nets = {}
for model in descendants(ArghModel):
    # change the names so argh can create parsers
    model.args.__name__ = model.__name__
    model_options.append(model.args)
    nets[model.__name__] = model
argh.add_commands(mlp_parser, model_options, namespace='model', namespace_kwargs={'dest': 'model'})
for model in ArghModel.__subclasses__():
    # change names back
    model.args.__name__ = 'args'


args = parser.parse_args()
args.task(args)
