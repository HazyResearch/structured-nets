import sys, os, datetime
sys.path.insert(0, '../../pytorch/')
sys.path.insert(0, '../../tensorflow/')

def optimize(dataset, params, seed=None):
    if params.torch:
        from optimize_torch import optimize_torch
        return optimize_torch(dataset, params, seed)
    else:
        from optimize_tf import optimize_tf
        return optimize_tf(dataset, params)
