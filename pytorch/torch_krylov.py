import torch
import functools
import numpy as np
from torch.autograd import Variable
import time

# Down shift
def Z_mult_fn(f, x):
    return torch.cat((f * x[-1], x[:-1]))

# Up shift
def Z_transpose_mult_fn(f, x):
    #print('x[1:]: ', x[1:])
    #print('f*x[0]: ', f*x[0])
    #return torch.cat((x[1:], torch.FloatTensor([f * x[0]])))
    return torch.cat((x[1:], f*x[0]))

# Diagonal multiplication
def diag_mult_fn(diag, x):
    return diag * x

# Circulant sparsity pattern operators
def circ_mult_fn(subdiag_f, x):
    # note: f corresponds to last element instead of first
    y = torch.cat((x[-1], x[:-1]))
    return y * subdiag_f

def circ_transpose_mult_fn(subdiag_f, x):
    # Circular shift
    y = torch.cat((x[1:], x[0]))

    # Scale by [v f]
    return y * subdiag_f

# Tridiagonal + corner operators
# TODO NEEDS FIX
def tridiag_transpose_mult_fn(subdiag_f, diag, supdiag, x):
    y = torch.cat((x[1:], x[0]))
    sub_result = y * subdiag_f
    z = Variable(torch.zeros(1).cuda())
    sup_result = torch.cat((z, x[:-1] * supdiag))
    diag_result = x*diag

    return sup_result + sub_result + diag_result
