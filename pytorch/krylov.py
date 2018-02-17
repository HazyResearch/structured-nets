import torch
import functools
import numpy as np
import time

def tridiag_corner_transpose_mult_fn(subdiag, diag, supdiag, f, x):
 	sub_result = x[1:] * subdiag
	sup_result = x[:n-1] * supdiag

	z = torch.FloatTensor([0])
	sup_result = torch.cat((z, sup_result), dim=0)
	sub_result = torch.cat((sub_result, z), dim=0)
	diag_result = x*diag

	z = torch.zeros(n-1)

	f_result = torch.cat((z, f*x[[0]]), dim=0)

	return sup_result + sub_result + diag_result + f_result

def circ_transpose_mult_fn(subdiag_f, x):
	# Circular shift
	y = torch.cat((x[1:], x[[0]]))
	
	# Scale by [v f]
	return y * subdiag_f

# Circular shift
def toeplitz_like_mult_fn(f, x):
	return torch.cat((f * x[[x.size(0) - 1]], x[:-1]))

# fn: takes as input vector, multiplies by matrix.
def krylov(fn, v, n):
	cols = [v]
	for _ in range(n-1):
	         v = fn(v)
	         cols.append(v)
	return torch.stack(cols, dim=-1)

def test_toeplitz(n):
	f = 1
	fn = functools.partial(toeplitz_like_mult_fn, f)
	v = torch.arange(1, n+1)
	return krylov(fn, v, n)

def test_tridiag_corner(n):
	f = 1

	diag = torch.arange(1,n+1)
	subdiag = torch.arange(1,n)
	supdiag = torch.arange(1,n)

	fn = functools.partial(tridiag_corner_transpose_mult_fn, subdiag, 
		diag, supdiag, f)

	v = torch.arange(1, n+1)

	return krylov(fn, v, n)

def test_circ_transpose_mult_fn(n):
	f = 1
	subdiag = torch.arange(1,n)

	f = torch.FloatTensor([f])

	subdiag_f = torch.cat((subdiag, f),dim=0)

	fn = functools.partial(circ_transpose_mult_fn, subdiag_f)

	v = torch.arange(1, n+1)
	return krylov(fn, v, n)

if __name__ == "__main__":
	n = 1000
	t1 = time.time()
	test_toeplitz(n)
	print 'Toeplitz time: ', time.time() - t1
	t2 = time.time()
	test_tridiag_corner(n)
	print 'Tridiag+corner time: ', time.time() - t2
	t3 = time.time()
	test_circ_transpose_mult_fn(n)
	print 'Circulant sparsity time: ', time.time() - t3