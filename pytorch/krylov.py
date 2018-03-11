import torch
import functools
import numpy as np
import time

# x: [batch_size] x n array.
def tridiag_corner_transpose_mult_fn(subdiag, diag, supdiag, f, x):
	batch_size = x.size(0)

 	sub_result = x[:, 1:] * subdiag

	sup_result = x[:, :n-1] * supdiag

	z = torch.zeros(batch_size, 1)

	sup_result = torch.cat((z, sup_result), dim=1)

	sub_result = torch.cat((sub_result, z), dim=1)
	diag_result = x*diag

	z = torch.zeros(batch_size, n-1)
	f_x0 = f*x[:, [0]]

	f_result = torch.cat((z, f_x0), dim=1)

	return sup_result + sub_result + diag_result + f_result


def circ_transpose_mult_fn(subdiag_f, x):
	# Circular shift
	y = torch.cat((x[:, 1:], x[:, [0]]), dim=1)
	
	# Scale by [v f]
	return y * subdiag_f

# Circular shift
def toeplitz_like_mult_fn(f, x):
	return torch.cat((f * x[:,[x.size(0) - 1]], x[:, :-1]), dim=1)

# fn: takes as input vector, multiplies by matrix.
# v: [batch_size] x n array.
def krylov(fn, v, n):
	K = torch.zeros(n, n)
	K[:, 0] = v
	idx = 1
	while idx <= n-1:
		K[:, idx] = fn(K[:, idx-1])
		idx += 1
	return K

# fn: takes as input vector, multiplies by matrix.
# v: [batch_size] x n array.
def krylov_slow(fn, v, n):
	cols = []
	for i in range(n-1):
		v = fn(v)
		cols.append(v)
	return torch.stack(cols, dim=-1)

def krylov_other(fn, v, n):
	K = tf.zeros((n, n))
	for i in range(n-1):
		v = fn(v)
		K[:, i] = v
	return K

def test_toeplitz(n):
	f = 1
	fn = functools.partial(toeplitz_like_mult_fn, f)
	xs = torch.ones(2, n)
	return krylov(fn, xs, n)

def test_tridiag_corner(n):
	f = 1

	diag = torch.arange(1,n+1)
	subdiag = torch.arange(1,n)
	supdiag = torch.arange(1,n)

	fn = functools.partial(tridiag_corner_transpose_mult_fn, subdiag, 
		diag, supdiag, f)

	xs = torch.ones(1, n)
	return krylov(fn, xs, n)

def test_circ_transpose_mult_fn(n):
	f = 1
	subdiag = torch.arange(1,n)

	f = torch.FloatTensor([f])

	subdiag_f = torch.cat((subdiag, f),dim=0)

	fn = functools.partial(circ_transpose_mult_fn, subdiag_f)

	xs = torch.ones(1, n)
	return krylov(fn, xs, n)


def test_batch_tridiag(n):
	subdiag = torch.arange(1, n)
	diag = torch.arange(1, n+1)
	supdiag = torch.arange(1, n)
	f = torch.FloatTensor([1])

	fn = functools.partial(batch_tridiag_corner_transpose_mult_fn, subdiag, 
		diag, supdiag, f)

	xs = torch.ones(2, n)
	return krylov(fn, xs, n)

if __name__ == "__main__":
	n = 1000
	trials = 50
	#t1 = time.time()
	#test_toeplitz(n)
	#print 'Toeplitz time: ', time.time() - t1
	#t2 = time.time()
	#test_tridiag_corner(n)
	#print 'Tridiag+corner time: ', time.time() - t2
	times = 0
	for _ in range(trials):
		t3 = time.time()
		test_circ_transpose_mult_fn(n)
		this_time = time.time() - t3
		times += this_time
		print 'Tridiag+corner time: ', this_time
	print 'Average time: ', times/trials