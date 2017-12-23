import numpy as np
import tensorflow as tf
from utils import *
from reconstruction import *

def vandermonde_like(M, test_X, test_Y, r, steps=10, batch_size=100, lr=0.002, mom=0.9, 
	test_freq=100, verbose=False):

	assert M.shape[0] == M.shape[1]
	n = M.shape[0]

	# A is learned, B is fixed
	B_vand = gen_Z_f(M.shape[0], 0).T
	f_V = 0

	# Create the model
	x = tf.placeholder(tf.float64, [None, n])
	v = tf.Variable(tf.truncated_normal([n], stddev=0.01, dtype=tf.float64))
	G = tf.Variable(tf.truncated_normal([n, r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([n, r], stddev=0.01, dtype=tf.float64))
	W1 = vand_recon(G, H, v, n, n, f_V, r)
	y = tf.matmul(x, W1)
	y_ = tf.placeholder(tf.float64, [None, n])
	mse = tf.reduce_mean(tf.squared_difference(y, y_))
	
	train_step = tf.train.MomentumOptimizer(lr, mom).minimize(mse)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	step = 0

	losses = []
	while step < steps:
		batch_xs, batch_ys = gen_batch(M, batch_size)
		_, = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
	 
		if step % test_freq == 0:
			print('Training step: ', step)
			# Verify displacement rank
			v_real, W1_real = sess.run([v, W1], feed_dict={x: batch_xs, y_: batch_ys})
			A = np.diag(v_real)
			E = W1_real - np.dot(A, np.dot(W1_real, B_vand))
			print('Disp rank: ', np.linalg.matrix_rank(E))
			this_loss = sess.run(mse, feed_dict={x: test_X, y_: test_Y})
			losses.append(this_loss)
			print('Test loss: ', this_loss)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, Vandermonde-like: ', sess.run(mse, feed_dict={x: test_X, y_: test_Y}))

	return losses

def hankel_like(M, test_X, test_Y, r, steps=10, batch_size=100, lr=0.1, mom=0.9, test_freq=100, verbose=False):
	f_H = 0
	g_H = 1

	n = M.shape[0]
	assert M.shape[0] == M.shape[0]

	A_hank = gen_Z_f(n, f_H)
	B_hank = gen_Z_f(n, g_H)

	# Check true DR
	E_M_stein = M - np.dot(A_hank, np.dot(M, B_hank))

	print('Disp rank of M, Stein: ', np.linalg.matrix_rank(E_M_stein))

	# Create the model
	x = tf.placeholder(tf.float64, [None, n])
	G = tf.Variable(tf.truncated_normal([n, r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([n, r], stddev=0.01, dtype=tf.float64))

	# Reconstruct W1 from G and H
	W1 = rect_recon_tf(G, H, B_hank, n, n, f_H, g_H, r)
	y = tf.matmul(x, W1)
	y_ = tf.placeholder(tf.float64, [None, n])
	mse = tf.reduce_mean(tf.squared_difference(y, y_))
	train_step = tf.train.MomentumOptimizer(lr, mom).minimize(mse)

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	step = 0

	losses = []
	while step < steps:
		batch_xs, batch_ys = gen_batch(M, batch_size)

		_, = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
	 
		if step % test_freq == 0:
			print('Training step: ', step)
			# Verify displacement rank
			W1_real = sess.run(W1, feed_dict={x: batch_xs, y_: batch_ys})
			
			E_stein = W1_real - np.dot(A_hank, np.dot(W1_real, B_hank))

			print('Disp rank, Stein: ', np.linalg.matrix_rank(E_stein))
			this_loss = sess.run(mse, feed_dict={x: test_X, y_: test_Y})
			losses.append(this_loss)
			print('Test loss: ', this_loss)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, Hankel-like: ', sess.run(mse, feed_dict={x: test_X, y_: test_Y}))

	return losses

def toeplitz_like(M, test_X, test_Y, r, steps=10, batch_size=100, lr=0.002, mom=0.9, 
	test_freq=100, verbose=False):

	A = gen_Z_f(M.shape[0], 1)
	B = gen_Z_f(M.shape[1], -1)

	n = A.shape[0]

	assert A.shape[0] == B.shape[0]

	# Check true DR
	E_M_sylv = np.dot(A,M) - np.dot(M, B)

	print('Disp rank of M, Sylv: ', np.linalg.matrix_rank(E_M_sylv))

	# Create the model
	x = tf.placeholder(tf.float64, [None, n])
	G = tf.Variable(tf.truncated_normal([n, r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([n, r], stddev=0.01, dtype=tf.float64))

	# Reconstruct W1 from G and H
	W1 = tf.zeros([n, n], dtype=tf.float64)
	f = 1
	g = -1
	f_mask = tf.constant([[f if j > k else 1 for j in range(n)] for k in range(n)], dtype=tf.float64)
	g_mask = tf.constant([[g if j > k else 1 for j in range(n)] for k in range(n)], dtype=tf.float64)
	index_arr = gen_index_arr(n)

	for i in range(r):
		Z_g_i = circulant_tf(G[:, i], index_arr, f_mask)
		Z_h_i = circulant_tf(tf.reverse(H[:, i], tf.constant([0])), index_arr, g_mask)
		prod = tf.matmul(Z_g_i, Z_h_i)
		W1 = tf.add(W1, prod)

	W1 = tf.scalar_mul(0.5, W1)
	y = tf.matmul(x, W1)
	y_ = tf.placeholder(tf.float64, [None, n])
	mse = tf.reduce_mean(tf.squared_difference(y, y_))
	train_step = tf.train.MomentumOptimizer(lr, mom).minimize(mse)

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	step = 0

	losses = []
	while step < steps:
		batch_xs, batch_ys = gen_batch(M, batch_size)

		_, = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
	 
		if step % test_freq == 0:
			# Verify displacement rank
			print('Training step: ', step)
			W1_real, g_real, h_real = sess.run([W1, Z_g_i, Z_h_i], feed_dict={x: batch_xs, y_: batch_ys})

			E_sylv = np.dot(A, W1_real) - np.dot(W1_real, B)

			print('Disp rank, Sylv: ', np.linalg.matrix_rank(E_sylv))
			this_loss = sess.run(mse, feed_dict={x: test_X, y_: test_Y})
			print('Test loss: ', this_loss)
			losses.append(this_loss)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, Toeplitz-like: ', sess.run(mse, feed_dict={x: test_X, y_: test_Y}))

	return losses