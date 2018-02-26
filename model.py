from scipy.sparse import diags
import numpy as np
import tensorflow as tf
import functools
from reconstruction import *
from utils import *
from krylov import *

def check_rank(sess, x, y_, batch_xs, batch_ys, params, model):
	if not params.check_disp:
		return 

	if params.class_type in ['unconstrained', 'tridiagonal_corner', 'symmetric']:
		if params.class_type == 'symmetric':
			A = sess.run(model['A'], feed_dict={x: batch_xs, y_: batch_ys})
			B = sess.run(model['B'], feed_dict={x: batch_xs, y_: batch_ys})

			print 'A: ', A
			print 'B: ', B

		return
	if params.class_type in ['toeplitz_like', 'hankel_like']:
		A,B = gen_operators(params)
		W = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
		print 'W: ', W.shape
		E = compute_disp(params.disp_type, W, A, B)
	elif params.class_type == 'circulant_sparsity':
		# Construct A
		f_x_A = sess.run(model['f_x_A'], feed_dict={x: batch_xs, y_: batch_ys})
		# Construct B
		f_x_B = sess.run(model['f_x_B'], feed_dict={x: batch_xs, y_: batch_ys})

		if params.fix_A_identity:
			A = np.eye(params.layer_size)
		else:
			A = gen_Z_f(params.layer_size, f_x_B[0], f_x_B[1:]).T

		B = gen_Z_f(params.layer_size, f_x_B[0], f_x_B[1:]).T

		W = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
		E = compute_disp(params.disp_type, W, A, B)
	elif params.class_type == 'tridiagonal_corner':
		# Construct A
		subdiag_A, supdiag_A, diag_A, f_A = sess.run([model['subdiag_A'], model['supdiag_A'], model['diag_A'], model['f_A']], 
			feed_dict={x: batch_xs, y_: batch_ys})
		A = gen_tridiag_corner(subdiag_A, supdiag_A, diag_A, f_A)
		# Construct B
		subdiag_B, supdiag_B, diag_B, f_B = sess.run([model['subdiag_B'], model['supdiag_B'], model['diag_B'], model['f_B']], 
			feed_dict={x: batch_xs, y_: batch_ys})		
		B = gen_tridiag_corner(subdiag_B, supdiag_B, diag_B, f_B)
		W = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
		E = compute_disp(params.disp_type, W, A, B)
	elif params.class_type == 'low_rank':
		E = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
	elif params.class_type == 'vandermonde_like':
		v, W = sess.run([model['v'], model['W']], feed_dict={x: batch_xs, y_: batch_ys})
		A = np.diag(v)
		B = gen_Z_f(params.layer_size, 0).T
		E = compute_disp(params.disp_type, W, A, B)
	else:
		print 'class_type not supported: ', params.class_type
		assert 0 
	print E.shape
	print('(Displacement) Rank: ', np.linalg.matrix_rank(E))

def get_structured_W(params):
	model = {}
	if params.class_type == 'unconstrained':
		W = tf.Variable(tf.truncated_normal([params.layer_size, params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
		model['W'] = W
		return W, model
	elif params.class_type in ['low_rank', 'symmetric', 'toeplitz_like', 
		'vandermonde_like', 'hankel_like', 'circulant_sparsity', 'tridiagonal_corner']:
		G = tf.Variable(tf.truncated_normal([params.layer_size, params.r], stddev=params.init_stddev, dtype=tf.float64))
		H = tf.Variable(tf.truncated_normal([params.layer_size, params.r], stddev=params.init_stddev, dtype=tf.float64))
		if params.class_type == 'low_rank':
			W = tf.matmul(G, tf.transpose(H))
		elif params.class_type == 'symmetric':
			# Initialization with T+H operators
			Z1 = gen_Z_f(params.layer_size, 1)
			Z1m = gen_Z_f(params.layer_size, -1)

			op_A = Z1 + Z1.T
			op_B = Z1m + Z1m.T

			#print 'op_A: ', op_A
			#print 'op_B: ', op_B

			op_A = np.random.random((params.layer_size, params.layer_size))
			op_B = np.random.random((params.layer_size, params.layer_size))

			A = tf.Variable(op_A)
			A_upper = tf.matrix_band_part(A, 0, -1)
			A_symm = 0.5 * (A_upper + tf.transpose(A_upper))

			B = tf.Variable(op_B)
			B_upper = tf.matrix_band_part(B, 0, -1)
			B_symm = 0.5 * (B_upper + tf.transpose(B_upper))

			W = general_recon(G, H, A_symm, B_symm)
			model['A'] = A_symm
			model['B'] = B_symm
		elif params.class_type == 'toeplitz_like':
			W = toeplitz_like_recon(G, H, params.layer_size, params.r)
		elif params.class_type == 'hankel_like':
			f = 0
			g = 1
			B = gen_Z_f(params.layer_size, g)
			W = rect_recon_tf(G, H, B, params.layer_size, params.layer_size, f, g, params.r)
		elif params.class_type == 'vandermonde_like':
			f_V = 0
			v = tf.Variable(tf.truncated_normal([params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
			model['v'] = v
			W = vand_recon(G, H, v, params.layer_size, params.layer_size, f_V, params.r)
		elif params.class_type == 'symmetric_tridiagonal':
			return 0
		elif params.class_type == 'circulant_sparsity':
			f_x_A, f_x_B = get_f_x(params.layer_size, params.init_type, params.learn_corner, params.n_diag_learned, params.init_stddev)

			if params.learn_diagonal:
				diag_A = tf.Variable(tf.zeros(params.layer_size, dtype=tf.float64))
				diag_B = tf.Variable(tf.zeros(params.layer_size, dtype=tf.float64))
				fn_A = functools.partial(circ_diag_transpose_mult_fn, tf.reverse(f_x_A, [0]), diag_A)
				fn_B = functools.partial(circ_diag_transpose_mult_fn, tf.reverse(f_x_B, [0]), diag_B)

			elif params.fix_A_identity:
				# f_x_A is unused
				print 'fixing A to be identity'
				fn_A = identity_mult_fn
				fn_B = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_B, [0]))

			else:
				fn_A = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_A, [0]))
				fn_B = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_B, [0]))

			W = krylov_recon(params, G, H, fn_A, fn_B)

			# Compute a and b
			a = tf.reduce_prod(f_x_A)
			b = tf.reduce_prod(f_x_B)

			coeff = 1.0/(1 - a*b)

			W = tf.scalar_mul(coeff, W)
			model['f_x_A'] = f_x_A
			model['f_x_B'] = f_x_B

		elif params.class_type == 'tridiagonal_corner':
			subdiag_A, supdiag_A, diag_A, f_A, subdiag_B, supdiag_B, diag_B, f_B = get_tridiag_corner_vars(params.layer_size, params.init_type, params.init_stddev, params.learn_corner)

			model['subdiag_A'] = subdiag_A
			model['supdiag_A'] = supdiag_A
			model['diag_A'] = diag_A
			model['f_A'] = f_A
			model['subdiag_B'] = subdiag_B
			model['supdiag_B'] = supdiag_B
			model['diag_B'] = diag_B
			model['f_B'] = f_B

			fn_A = functools.partial(tridiag_corner_transpose_mult_fn, subdiag_A, diag_A, supdiag_A, f_A)
			fn_B = functools.partial(tridiag_corner_transpose_mult_fn, subdiag_B, diag_B, supdiag_B, f_B)
			W = krylov_recon(params, G, H, fn_A, fn_B)
			# Compute a and b
			a = tf.multiply(f_A, tf.reduce_prod(subdiag_A))
			b = tf.multiply(f_B, tf.reduce_prod(subdiag_B))

			coeff = 1.0/(1 - a*b)

			W = tf.multiply(coeff, W)
		model['W'] = W
		return W, model


	else:
		print 'Not supported: ', params.class_type	
		assert 0	

def forward(x, params):
 	W, model = get_structured_W(params)
	y = compute_y(x, W, params)
	return y, model

def compute_y(x, W1, params):
	if params.transform == 'cnn':
		return compute_y_cnn(x, W1, params)
	elif params.num_layers==0:
		y = tf.matmul(x, W1)
		return y
	elif params.num_layers==1:
		b1 = tf.Variable(tf.truncated_normal([params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
		W2 = tf.Variable(tf.truncated_normal([params.layer_size, params.out_size], stddev=params.init_stddev, dtype=tf.float64))
		b2 = tf.Variable(tf.truncated_normal([params.out_size], stddev=params.init_stddev, dtype=tf.float64))
		xW = tf.matmul(x, W1)

		h = tf.nn.relu(xW + b1)
		y = tf.matmul(h, W2) + b2
		return y
	else:
		print 'Not supported: ', params.num_layers
		assert 0