from scipy.linalg import toeplitz, circulant, solve_sylvester
from scipy.sparse import diags
import numpy as np
import tensorflow as tf
import time

def gen_init_operators(init, m, n):
	if init == 'toeplitz':
		A = tf.Variable(gen_Z_f(m, 1))
		B = tf.Variable(gen_Z_f(n, -1))
	elif init == 'hankel':
		A = tf.Variable(gen_Z_f(m, 1))
		B = tf.Variable(gen_Z_f(n, 0).T)
	elif init == 't+h':
		A = tf.Variable(gen_Z_f(m, 0) + gen_Z_f(m, 0).T)
		B = tf.Variable(gen_Z_f(n, 0) + gen_Z_f(n, 0).T)
	elif init == 'vandermonde':
		v = np.random.random(n)
		A = tf.Variable(np.diag(v))
		B = tf.Variable(gen_Z_f(n, 0))
	elif init == 'random':
		A = tf.Variable(tf.truncated_normal([m, m], stddev=0.01, dtype=tf.float64))
		B = tf.Variable(tf.truncated_normal([n, n], stddev=0.01, dtype=tf.float64))	
	else:
		print 'Init type ' + prefix + ' not supported'
		assert 0
	return A,B

def gen_trid_mask(n):
	ones1 = list(np.ones(n))
	ones2 = list(np.ones(n-1))
	data = [ones1, ones2, ones2]
	positions = [0, 1, -1]

	mask = diags(data, positions, (n, n)).toarray()

	return mask

def gen_trid_corner_mask(n):
	mask = gen_trid_mask(n)
	mask[0, -1] = 1

	return mask

def gen_Z_f(m, f):
	I_m = np.eye(m-1, m-1)
	Z_f = np.hstack((I_m, np.zeros((m-1, 1))))
	Z_f = np.vstack((np.zeros((1, m)), Z_f)) 
	Z_f[0, -1] = f
	return Z_f

def gen_f_mask(f, m,n):
	mask = np.ones((m, n))
	f_mask = f*mask

	# Set lower triangular indices to 1: 1 if row>=col
	il1 = np.tril_indices(n=f_mask.shape[0], m=f_mask.shape[1])
	f_mask[il1] = 1 
	f_mask = tf.constant(f_mask, dtype=tf.float64)

	return f_mask

def gen_index_arr(n):
	r = np.arange(0, n)
	a = np.expand_dims(r, -1)
	b = np.expand_dims(-r, 0)

	index = a+b

	# Create mask
	pos = (index < 0)

	antipos = 1-pos

	updated = np.multiply(antipos, index) + np.multiply(pos, index+n)

	return np.expand_dims(updated, -1)

def sylvester(A, B, n, r):
	# Generate random rank r error matrix
	G = np.random.random((n, r))
	H = np.random.random((n, r))
	GH = np.dot(G,H.T)

	# Solve Sylvester equation to recover M
	# Such that AM - MB = GH^T
	M = solve_sylvester(A, -B, GH)

	E = np.dot(A,M) - np.dot(M,B)

	assert np.linalg.norm(E - GH) <= 1e-10

	return M

def gen_matrix(n, prefix, r=2):
	if prefix == 'toeplitz':
		c = np.random.random(n)
		r = np.random.random(n)
		return toeplitz(c,r)
	elif prefix == 'hankel':
		c = np.random.random(n)
		r = np.random.random(n)
		return np.flipud(toeplitz(c,r))	
	elif prefix == 'vandermonde':
		v = np.random.random(n)
		return np.vander(v, n, increasing=True)
	elif prefix == 'cauchy':
		return gen_cauchy(n)
	elif prefix == 'random':
		return np.random.random((n,n))
	elif prefix == 'toeplitz-like':
		# Generate operators
		A = gen_Z_f(n, 1)
		B = gen_Z_f(n, -1)
		# Generate random rank r error matrix
		# Solve sylvester
		return sylvester(A, B, n, r)
	elif prefix == 'hankel-like':
		# Generate operators
		A = gen_Z_f(n, 1)
		B = gen_Z_f(n, 0).T
		return sylvester(A, B, n, r)
	elif prefix == 'vandermonde-like':
		# Generate operators
		v = np.random.random(n)
		#v = np.linalg.eigvals(Z1)
		V = np.vander(v, increasing=True)

		# Generate operators
		A = np.diag(v)
		B = gen_Z_f(n, 0)

		return sylvester(A, B, n, r)

	elif prefix == 'cauchy-like':
		s = np.random.random(n)
		t = np.random.random(n)

		C = 1.0 / (s.reshape((-1,1)) - t)

		# Generate operators
		A = np.diag(s)
		B = np.diag(t)

		return sylvester(A, B, n, r)

	elif prefix == 'tridiag_corner': # 
		# Generate random tridiagonal+corner operators
		A = np.random.random((n,n))
		B = np.random.random((n,n))
		mask = gen_trid_corner_mask(n)
		A = np.multiply(A, mask)
		B = np.multiply(B, mask)

		return sylvester(A, B, n, r)
	elif prefix == 'circ_sparsity':
		# Generate random circulant sparsity pattern operators
		A = np.random.random((n,n))
		B = np.random.random((n,n))
		mask = gen_Z_f(n, 1)
		A = np.multiply(A, mask)
		B = np.multiply(B, mask)

		return sylvester(A, B, n, r)
	else:
		print 'Type ' + prefix + ' not supported'
		assert 0


def gen_batch(A, N):
	"""
	Generates N random x's, computes corresponding y's, such that Ax = y.
	A: the matrix.
	N: number of datapoints.
	"""

	X = np.random.random((A.shape[1], N))
	Y = np.dot(A, X)

	assert np.isclose(np.linalg.norm(Y[:, 0] - np.dot(A, X[:, 0])), 0)

	return X.T,Y.T

def get_GH(E):
	disp_rank = np.linalg.matrix_rank(E)

	# SVD
	U, S, V = np.linalg.svd(E, full_matrices=False)

	SV = np.dot(np.diag(S), V)
	G = U[:, 0:disp_rank]
	H = SV[0:disp_rank, :].T

	return G,H, disp_rank

def sylvester_project(M, A, B, r):
	"""
	Project via SVD on error matrix + solving the Sylvester equation.
	"""
	t = time.time()
	E = np.dot(A, M) - np.dot(M, B)
	G,H,dr = get_GH(E)

	G_r = G[:, 0:r]
	H_r = H[:, 0:r]

	lowrank = np.dot(G_r, H_r.T)

	print 'norm(E-lowrank): ', np.linalg.norm(E-lowrank)

	# Sylvester solve
	M_class = solve_sylvester(A, -B, lowrank)

	print 'rank(lowrank): ', np.linalg.matrix_rank(lowrank)
	print 'rank(A): ', np.linalg.matrix_rank(A)
	print 'rank(B): ', np.linalg.matrix_rank(B)
	print 'norm(M-M_class): ', np.linalg.norm(M-M_class)

	E_class = np.dot(A, M_class) - np.dot(M_class, B)
	print 'rank of E_class',np.linalg.matrix_rank(E_class)
	#print 'eigvals of E_class',np.linalg.eigvals(E_class)

	print 'time of sylv project: ', time.time() - t

	return M_class

def circulant_tf(vec, index_arr, mask=None):
	# Slice 2D
	output = tf.gather_nd(vec, index_arr)

	# Apply mask
	if mask is not None:
		output = tf.multiply(output, mask)

	return output

# Shape of stack_circ: (v.size, n)
def circulant_mn_tf(v, index_arr, n, num_reps, f_mask):
	circ_v = circulant_tf(v, index_arr)

	multiples = tf.constant([1, num_reps]) 

	stack_circ = tf.tile(circ_v, multiples)
	stack_circ = tf.cast(stack_circ[:, 0:n], tf.float64)

	# Element-wise multiplication
	masked = tf.multiply(f_mask, stack_circ)

	return masked  

def krylov_tf(A, v, n):
	v_exp = tf.expand_dims(v,1)
	cols = [v_exp]
	this_pow = A

	for i in range(n-1):
		this_col = tf.matmul(this_pow, v_exp)

		this_pow = tf.matmul(A, this_pow)

		cols.append(this_col)

		K = tf.stack(cols)
	return tf.squeeze(K)


def V_mn(v, m, n):
	# Stack columns
	# First col: ones
	# Second col: v
	# Subsequent cols: v^{c-1}
	ones = tf.ones([m], dtype=tf.float64)

	cols = [ones, v]

	for i in range(n-2):
		this_col = tf.pow(v, i+2)
		cols.append(this_col)

	V = tf.transpose(tf.stack(cols))

	return tf.cast(V, tf.float64)