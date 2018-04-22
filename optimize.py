import sys, os, datetime
sys.path.insert(0, '../../pytorch/')
sys.path.insert(0, '../../tensorflow/')
from optimize_tf import optimize_tf
from optimize_torch import optimize_torch

def optimize(dataset, params):
	if params.torch:
		return optimize_torch(dataset, params)
	else:
		return optimize_tf(dataset, params)
