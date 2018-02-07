import numpy as np
import tensorflow as tf
from utils import *
from reconstruction import *

def vandermonde_like(dataset, params, test_freq=100, verbose=False):
	# A is learned, B is fixed
	B_vand = gen_Z_f(params.n, 0).T
	f_V = 0

	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	v = tf.Variable(tf.truncated_normal([params.n], stddev=0.01, dtype=tf.float64))
	G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	W1 = vand_recon(G, H, v, params.n, params.n, f_V, params.r)
	
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
		_, = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
	 
		if step % test_freq == 0:
			print('Training step: ', step)
			# Verify displacement rank
			v_real, W1_real = sess.run([v, W1], feed_dict={x: batch_xs, y_: batch_ys})
			A = np.diag(v_real)
			E = W1_real - np.dot(A, np.dot(W1_real, B_vand))
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
	print('SGD final loss, Vandermonde-like: ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, Vandermonde-like: ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies

def hankel_like(dataset, params, test_freq=100, verbose=False):
	f = 0
	g = 1
	A = gen_Z_f(params.n, f)
	B = gen_Z_f(params.n, g)

	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	v = tf.Variable(tf.truncated_normal([params.n], stddev=0.01, dtype=tf.float64))
	G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))

	W1 = rect_recon_tf(G, H, B, params.n, params.n, f, g, params.r)

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
		_ = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
 
		if step % test_freq == 0:
			print('Training step: ', step)
			# Verify displacement rank
			W1_real = sess.run(W1, feed_dict={x: batch_xs, y_: batch_ys})
			E = W1_real - np.dot(A, np.dot(W1_real, B))
			print('Disp rank: ', np.linalg.matrix_rank(E))
			this_loss, this_accuracy, summary = sess.run([loss, accuracy, merged_summary_op], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

			summary_writer.add_summary(summary, step)
			losses.append(this_loss)
			accuracies.append(this_accuracy)
			print('Test loss: ', this_loss)
			print('Test accuracy: ', this_accuracy)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, Hankel-like: ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, Hankel-like: ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies


def toeplitz_like(dataset, params, test_freq=100, verbose=False):
	A = gen_Z_f(params.n, 1)
	B = gen_Z_f(params.n, -1)

	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))

	W1 = toeplitz_like_recon(G, H, params.n, params.r)

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
		_, = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

	 
		if step % test_freq == 0:
			print('Training step: ', step)
			
			# Verify displacement rank
			W1_real = sess.run(W1, feed_dict={x: batch_xs, y_: batch_ys})
			E_sylv = np.dot(A, W1_real) - np.dot(W1_real, B)
			print('Disp rank, Sylv: ', np.linalg.matrix_rank(E_sylv))

			this_loss, this_accuracy, summary = sess.run([loss, accuracy, merged_summary_op], feed_dict={x: dataset.test_X, y_: dataset.test_Y})
			summary_writer.add_summary(summary, step)
			losses.append(this_loss)
			accuracies.append(this_accuracy)
			print('Test loss: ', this_loss)
			print('Test accuracy: ', this_accuracy)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, Toeplitz-like: ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, Toeplitz-like: ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies


def low_rank(dataset, params, test_freq=100, verbose=False):
	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.r, params.n], stddev=0.01, dtype=tf.float64))
	W1 = tf.matmul(G, H)

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
		_, = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

	 
		if step % test_freq == 0:
			print('Training step: ', step)
			this_loss, this_accuracy, summary = sess.run([loss, accuracy, merged_summary_op], feed_dict={x: dataset.test_X, y_: dataset.test_Y})
			summary_writer.add_summary(summary, step)
			losses.append(this_loss)
			accuracies.append(this_accuracy)
			print('Test loss: ', this_loss)
			print('Test accuracy: ', this_accuracy)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, low rank: ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, low rank: ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies

def unconstrained(dataset, params, test_freq=100, verbose=False):
	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	W1 = tf.Variable(tf.truncated_normal([params.n, params.n], stddev=0.01, dtype=tf.float64))
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
		_ = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

	 
		if step % test_freq == 0:
			print('Training step: ', step)
			this_loss, this_accuracy, summary = sess.run([loss, accuracy, merged_summary_op], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

			summary_writer.add_summary(summary, step)
			losses.append(this_loss)
			accuracies.append(this_accuracy)
			print('Test loss: ', this_loss)
			print('Test accuracy: ', this_accuracy)
			if verbose:
				print('Current W1: ', W1_real)

		step += 1

	# Test trained model
	print('SGD final loss, unconstrained: ', sess.run(loss, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))
	print('SGD final accuracy, unconstrained: ', sess.run(accuracy, feed_dict={x: dataset.test_X, y_: dataset.test_Y}))

	return losses, accuracies

