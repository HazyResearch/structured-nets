import sys
sys.path.insert(0, '../')
from reconstruction import *
from model_params import ModelParams
from utils import *
from krylov import *
from scipy.linalg import toeplitz
import numpy as np

def test_circ_sparsity(n):
	# Generate Toeplitz matrix
	A = gen_Z_f(n, 1).T
	B = gen_Z_f(n, -1)

	M = toeplitz(np.random.random(n), np.random.random(n))

	# Stein displacement
	E = M - np.dot(np.dot(A,M), B)

	G,H,r = get_GH(E)

	G = tf.constant(G, dtype=tf.float64)
	H = tf.constant(H, dtype=tf.float64)

	print r

	f_x_A, f_x_B = get_f_x(n, 'toeplitz', True, n-1)

	fn_A = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_A, [0]))
	fn_B = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_B, [0]))

	W1 = tf.zeros([n, n], dtype=tf.float64)
	for i in range(r):
		K_A = krylov(fn_A, G[:, i], n)
		K_B = krylov(fn_B, H[:, i], n)
		prod = tf.matmul(K_A, tf.transpose(K_B))
		W1 = tf.add(W1, prod)

	# Compute a and b
	a = tf.reduce_prod(f_x_A)
	b = tf.reduce_prod(f_x_B)

	coeff = 1.0/(1 - a*b)

	W1 = tf.scalar_mul(coeff, W1)

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	W1_real = sess.run(W1)

	print np.linalg.norm(W1_real - M)

def test_tridiag_corner(n):
	# Generate Toeplitz matrix
	A = gen_Z_f(n, 1).T
	B = gen_Z_f(n, -1)

	M = toeplitz(np.random.random(n), np.random.random(n))

	# Stein displacement
	E = M - np.dot(np.dot(A,M), B)

	G,H,r = get_GH(E)

	print r

	subdiag_A, supdiag_A, diag_A, f_A, subdiag_B, supdiag_B, diag_B, f_B = get_tridiag_corner_vars(n, 'toeplitz')

	fn_A = functools.partial(tridiag_corner_transpose_mult_fn, subdiag_A, diag_A, supdiag_A, f_A)
	fn_B = functools.partial(tridiag_corner_transpose_mult_fn, subdiag_B, diag_B, supdiag_B, f_B)

	W1 = tf.zeros([n, n], dtype=tf.float64)
	for i in range(r):
		K_A = krylov(fn_A, G[:, i], n)
		K_B = krylov(fn_B, H[:, i], n)
		prod = tf.matmul(K_A, tf.transpose(K_B))
		W1 = tf.add(W1, prod)

	# Compute a and b
	a = tf.multiply(f_A, tf.reduce_prod(subdiag_A))
	b = tf.multiply(f_B, tf.reduce_prod(subdiag_B))

	coeff = 1.0/(1 - a*b)

	W1 = tf.multiply(coeff, W1)

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	W1_real = sess.run(W1)

	print np.linalg.norm(W1_real - M)

def test_toeplitz(n):
	# Check that a Toeplitz matrix can be reconstructed
	A = gen_Z_f(n, 1).T
	B = gen_Z_f(n, -1)

	M = toeplitz(np.random.random(n), np.random.random(n))

	# Stein displacement
	E = M - np.dot(np.dot(A,M), B)

	G,H,r = get_GH(E)

	print r

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	M_recon = sess.run(general_tf(A, B, G, H, r, n, n))

	print M_recon
	print M
	print np.linalg.norm(M_recon - M)

def test_krylov(n):
	A = gen_Z_f(n, 1)
	v = tf.constant(np.arange(n), dtype=tf.float64)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	K = sess.run(krylov_tf(A, v, n))
	print K

def run_tests(n):
	# TODO: add more
	test_toeplitz(n)
	test_krylov(n)
	test_tridiag_corner(n)
	test_circ_sparsity(n)

if __name__ == '__main__':
	run_tests(10)