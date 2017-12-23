"""
Learning displacement operators.
"""
import numpy as np
import tensorflow as tf
from utils import *
from reconstruction import *

np.random.seed(0)

# Unconstrained SGD
def unconstrained(M, test_X, test_Y, r, steps=10, batch_size=100, lr=0.002, mom=0.9, test_freq=100, verbose=False):
	m = M.shape[0]
	n = M.shape[1]

	# Create the model
	x = tf.placeholder(tf.float64, [None, n])
	W1 = tf.Variable(tf.truncated_normal([m, n], stddev=0.01, dtype=tf.float64))
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
			this_loss = sess.run(mse, feed_dict={x: test_X, y_: test_Y})
			losses.append(this_loss)
			print('Test loss: ', this_loss)
			if verbose:
				print('Current W1: ', sess.run([W1], feed_dict={x: batch_xs, y_: batch_ys}))

		step += 1

	print('SGD final loss, unconstrained: ', sess.run(mse, feed_dict={x: test_X, y_: test_Y}))

	return losses


# Enforce form of operators via A_mask and B_mask
def learn_operators(M, A_mask, B_mask, test_X, test_Y, r, steps=10, batch_size=100,
	lr=0.002, mom=0.9, init='toeplitz', test_freq=100, verbose=False):
	m = M.shape[0]
	n = M.shape[1]

	# Create the model
	x = tf.placeholder(tf.float64, [None, n])
	G = tf.Variable(tf.truncated_normal([m, r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([n, r], stddev=0.01, dtype=tf.float64))

	if init == 'toeplitz':
		T_A = gen_Z_f(m, -1)
		T_B = gen_Z_f(n, 1)
		A = tf.Variable(T_A)
		B = tf.Variable(T_B)
	else:
		A = tf.Variable(tf.truncated_normal([m, m], stddev=0.01, dtype=tf.float64))
		B = tf.Variable(tf.truncated_normal([n, n], stddev=0.01, dtype=tf.float64))	
	
	# Enforce mask
	A = tf.multiply(A, A_mask)
	B = tf.multiply(B, B_mask)

	# Reconstruct W1 from G,H,A,B
	W1 = general_tf(A, B, G, H, r, m, n)

	y = tf.matmul(x, W1)

	# Define loss and optimizer
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
			# Check displacement rank
			W1_real, A_real, B_real = sess.run([W1, A, B], feed_dict={x: batch_xs, y_: batch_ys})
			E = np.dot(A_real, W1_real) - np.dot(W1_real, B_real)
			
			print 'Rank of error matrix: ', np.linalg.matrix_rank(E)

			this_loss = sess.run(mse, feed_dict={x: test_X, y_: test_Y})
			losses.append(this_loss)
			print('Test loss: ', this_loss)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1


	# Test trained model
	mse = sess.run(mse, feed_dict={x: test_X, y_: test_Y})

	print('SGD final loss, learned operators: ', mse)

	# Project onto class via SVD + Sylvester solver
	W1_learned, A_learned, B_learned = sess.run([W1, A, B], feed_dict={x: batch_xs, y_: batch_ys})
	W1_class = sylvester_project(W1_learned, A_learned, B_learned, r)
	y_class = tf.matmul(x, tf.constant(W1_class))

	mse_class = tf.reduce_mean(tf.squared_difference(y_class, y_))
	print 'Final loss after projection: ', sess.run(mse_class, feed_dict={x: test_X, y_: test_Y})

	return losses

def tridiagonal_corner(M, test_X, test_Y, r, steps, batch_size, lr, mom):
	# Generate tridiagonal+corner mask
	A_mask = tf.constant(gen_trid_corner_mask(M.shape[0]))
	B_mask = tf.constant(gen_trid_corner_mask(M.shape[1]))
	return learn_operators(M, A_mask, B_mask, test_X, test_Y, r, steps, batch_size, lr, mom)

def circulant_sparsity(M, test_X, test_Y, r, steps, batch_size, lr, mom):
	# General circulant sparsity pattern mask
	A_mask = tf.constant(gen_Z_f(M.shape[0], 1))
	B_mask = tf.constant(gen_Z_f(M.shape[1], 1))
	return learn_operators(M, A_mask, B_mask, test_X, test_Y, r, steps, batch_size, lr, mom)
