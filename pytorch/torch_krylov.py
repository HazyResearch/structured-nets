import torch
import functools
import numpy as np
from torch.autograd import Variable
import time

# Down shift
def Z_mult_fn(f, x):
	return torch.cat((f * x[0], x[1:]))

# Up shift
def Z_transpose_mult_fn(f, x):
	return torch.cat((x[1:], f * x[0]))

# Diagonal multiplication
def diag_mult_fn(diag, x):
	return diag * x

# Circulant sparsity pattern operators
def circ_transpose_mult_fn(subdiag_f, x):
	# Circular shift
	y = torch.cat((x[1:], x[0]))
	
	# Scale by [v f]
	return y * subdiag_f

# Tridiagonal + corner operators
def tridiag_transpose_mult_fn(subdiag_f, diag, supdiag, x):
	y = torch.cat((x[1:], x[0]))
	sub_result = y * subdiag_f	
	z = Variable(torch.zeros(1).cuda())
	sup_result = torch.cat((z, x[:-1] * supdiag))
	diag_result = x*diag

	return sup_result + sub_result + diag_result

