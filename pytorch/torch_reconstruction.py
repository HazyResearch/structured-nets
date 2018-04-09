import torch
from torch.autograd import Variable
import time
from torch_utils import *
from torch_krylov import *
from scipy.linalg import toeplitz
import numpy as np
import functools

def krylov(fn, v, n):
    cols = [v]
    for _ in range(n - 1):
        v = fn(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)

def krylov_recon(r, n, G, H, fn_A, fn_B_T):
	W1 = Variable(torch.zeros(n, n).cuda())

	for i in range(r):
		K_A = krylov(fn_A, G[:, i], n)
		
		K_B = krylov(fn_B_T, H[:, i], n).t()

		prod = torch.matmul(K_A, K_B).cuda()
		#print('W1: ', W1)
		#print('prod: ', prod)
		W1 += prod

	return W1

def recon(net):
	W = krylov_recon(net.params.r, net.params.layer_size, net.G, net.H, net.fn_A, net.fn_B_T)

	# Normalize
	if net.params.class_type == 'toeplitz_like':
		return 0.5*W 
	elif net.params.class_type in ['circulant_sparsity', 'tridiagonal_corner']:
		# Compute a and b
		a = torch.prod(net.subdiag_f_A)
		b = torch.prod(net.subdiag_f_B)

		coeff = 1.0/(1 - a*b)
		return coeff*W
	else:
		print('Class_type not supported: ', net.params.class_type)
		assert 0

if __name__ == '__main__':
	# Tests
	# Toeplitz matrix
	n = 10
	disp_rank = 2
	c = np.random.random(n)
	r = np.random.random(n)
	T = toeplitz(c,r)

	A = gen_Z_f(n, 1).T
	B = gen_Z_f(n, -1)
	E = T - np.dot(A,np.dot(T,B))
	print np.linalg.matrix_rank(E)

	U, S, V = np.linalg.svd(E, full_matrices=False)

	SV = np.dot(np.diag(S), V)
	G = U[:, 0:disp_rank]
	H = SV[0:disp_rank, :].T
	fn_A = functools.partial(Z_transpose_mult_fn, 1)
	fn_B_T = functools.partial(Z_transpose_mult_fn, -1)

	W = 0.5*krylov_recon(disp_rank, n, G, H, fn_A, fn_B_T)
	print 'W: ', W
	print 'T: ', T
