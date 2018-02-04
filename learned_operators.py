import numpy as np
import tensorflow as tf
from utils import *
from reconstruction import *
from krylov import *
import functools
import time

# Only an approximate reconstruction.
def tridiagonal_corner(dataset, params, test_freq=100, verbose=False):
	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	if params.fix_G:
		G = tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64)
	else:
		G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))

	subdiag_A, supdiag_A, diag_A, f_A, subdiag_B, supdiag_B, diag_B, f_B = get_tridiag_corner_vars(params.n, params.init_type, params.init_stddev, params.learn_corner)

	fn_A = functools.partial(tridiag_corner_transpose_mult_fn, subdiag_A, diag_A, supdiag_A, f_A)
	fn_B = functools.partial(tridiag_corner_transpose_mult_fn, subdiag_B, diag_B, supdiag_B, f_B)

	W1 = tf.zeros([params.n, params.n], dtype=tf.float64)
	for i in range(params.r):
		K_A = krylov(fn_A, G[:, i], params.n)
		K_B = krylov(fn_B, H[:, i], params.n)
		prod = tf.matmul(K_A, tf.transpose(K_B))
		W1 = tf.add(W1, prod)

	# Compute a and b
	a = tf.multiply(f_A, tf.reduce_prod(subdiag_A))
	b = tf.multiply(f_B, tf.reduce_prod(subdiag_B))

	coeff = 1.0/(1 - a*b)

	W1 = tf.multiply(coeff, W1)

	y = compute_y(x, W1, params)
	
	y_ = tf.placeholder(tf.float64, [None, params.out_size])
	
	loss, accuracy = compute_loss_and_accuracy(y, y_, params)
	
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', accuracy)

	merged_summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	step = 0

	losses = []
	accuracies = []
	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size)
		summary, _, = sess.run([merged_summary_op, train_step], feed_dict={x: batch_xs, y_: batch_ys})

		summary_writer.add_summary(summary, step)

		if step % test_freq == 0:
			print('Training step: ', step)
			this_loss, this_accuracy = sess.run([loss, accuracy], feed_dict={x: dataset.test_X, y_: dataset.test_Y})
			losses.append(this_loss)
			accuracies.append(this_accuracy)
			print('Test loss: ', this_loss)
			print('Test accuracy: ', this_accuracy)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, learned operators (OP transforms): ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, learned operators (OP transforms): ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies

def OP_transform(dataset, params, test_freq=100, verbose=False):
	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	if params.fix_G:
		G = tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64)
	else:
		G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))

	f_x_A, f_x_B = get_f_x(params)

	fn_A = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_A, [0]))
	fn_B = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_B, [0]))

	W1 = tf.zeros([params.n, params.n], dtype=tf.float64)
	for i in range(params.r):
		K_A = krylov(fn_A, G[:, i], params.n)
		K_B = krylov(fn_B, H[:, i], params.n)
		prod = tf.matmul(K_A, tf.transpose(K_B))
		W1 = tf.add(W1, prod)

	# Compute a and b

	a = tf.reduce_prod(f_x_A)
	b = tf.reduce_prod(f_x_B)

	coeff = 1.0/(1 - a*b)

	W1 = tf.scalar_mul(coeff, W1)

	y = compute_y(x, W1, params)
	
	y_ = tf.placeholder(tf.float64, [None, params.out_size])
	
	loss, accuracy = compute_loss_and_accuracy(y, y_, params)
	
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', accuracy)

	merged_summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	step = 0

	losses = []
	accuracies = []
	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size)
		summary, _, = sess.run([merged_summary_op, train_step], feed_dict={x: batch_xs, y_: batch_ys})

		summary_writer.add_summary(summary, step)

		if step % test_freq == 0:
			print('Training step: ', step)
			this_loss, this_accuracy = sess.run([loss, accuracy], feed_dict={x: dataset.test_X, y_: dataset.test_Y})
			losses.append(this_loss)
			accuracies.append(this_accuracy)
			print('Test loss: ', this_loss)
			print('Test accuracy: ', this_accuracy)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, learned operators (OP transforms): ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, learned operators (OP transforms): ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies

def circulant_sparsity(dataset, params, test_freq=100, verbose=False):
	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	if params.fix_G:
		G = tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64)
	else:
		G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))

	f_x_A, f_x_B = get_f_x(params.n, params.init_type, params.learn_corner, params.n_diag_learned, params.init_stddev)

	fn_A = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_A, [0]))
	fn_B = functools.partial(circ_transpose_mult_fn, tf.reverse(f_x_B, [0]))

	W1 = tf.zeros([params.n, params.n], dtype=tf.float64)
	for i in range(params.r):
		K_A = krylov(fn_A, G[:, i], params.n)
		K_B = krylov(fn_B, H[:, i], params.n)
		prod = tf.matmul(K_A, tf.transpose(K_B))
		W1 = tf.add(W1, prod)

	# Compute a and b
	a = tf.reduce_prod(f_x_A)
	b = tf.reduce_prod(f_x_B)

	coeff = 1.0/(1 - a*b)

	W1 = tf.scalar_mul(coeff, W1)

	y = compute_y(x, W1, params)
	
	y_ = tf.placeholder(tf.float64, [None, params.out_size])
	
	loss, accuracy = compute_loss_and_accuracy(y, y_, params)
	
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', accuracy)

	merged_summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	step = 0

	losses = []
	accuracies = []
	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size)
		summary, _, = sess.run([merged_summary_op, train_step], feed_dict={x: batch_xs, y_: batch_ys})

		summary_writer.add_summary(summary, step)

		if step % test_freq == 0:
			print('Training step: ', step)
			this_loss, this_accuracy = sess.run([loss, accuracy], feed_dict={x: dataset.test_X, y_: dataset.test_Y})
			losses.append(this_loss)
			accuracies.append(this_accuracy)
			print('Test loss: ', this_loss)
			print('Test accuracy: ', this_accuracy)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, learned operators (fixed circulant sparsity pattern): ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, learned operators (fixed circulant sparsity pattern): ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies

def circulant_sparsity_hadamard(dataset, params, test_freq=100, verbose=False):
	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	if params.fix_G:
		G = tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64)
	else:
		G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))

	t1 = time.time()
	W1, f_A, f_B, v_A, v_B = circ_sparsity_recon(G, H, params.n, params.r, params.learn_corner, 
		params.n_diag_learned, params.init_type, params.init_stddev)
	print 'overall time of circ_sparsity_recon: ', time.time() - t1

	y = compute_y(x, W1, params)
	
	y_ = tf.placeholder(tf.float64, [None, params.out_size])
	
	loss, accuracy = compute_loss_and_accuracy(y, y_, params)
	
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', accuracy)

	merged_summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	step = 0

	losses = []
	accuracies = []
	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size)
		summary, _, = sess.run([merged_summary_op, train_step], feed_dict={x: batch_xs, y_: batch_ys})

		summary_writer.add_summary(summary, step)
	 
		if step % test_freq == 0:
			print('Training step: ', step)
			# Verify displacement rank: Stein
			v_A_real = None
			v_B_real = None
			if params.n_diag_learned > 0:
				f_A_real, f_B_real, v_A_real, v_B_real, W1_real = sess.run([f_A, f_B, v_A, v_B, W1], feed_dict={x: batch_xs, y_: batch_ys})
			else:
				f_A_real, f_B_real, W1_real = sess.run([f_A, f_B, W1], feed_dict={x: batch_xs, y_: batch_ys})

			A = gen_Z_f(params.n, f_A_real, v_A_real).T
			B = gen_Z_f(params.n, f_B_real, v_B_real)

			E = W1_real - np.dot(A, np.dot(W1_real, B))
			print('Disp rank: ', np.linalg.matrix_rank(E))

			this_loss, this_accuracy = sess.run([loss, accuracy], feed_dict={x: dataset.test_X, y_: dataset.test_Y})
			losses.append(this_loss)
			accuracies.append(this_accuracy)
			print('Test loss: ', this_loss)
			print('Test accuracy: ', this_accuracy)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, learned operators (fixed circulant sparsity pattern): ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, learned operators (fixed circulant sparsity pattern): ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies
