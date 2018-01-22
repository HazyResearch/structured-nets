import numpy as np
import tensorflow as tf
from utils import *
from reconstruction import *

def circulant_sparsity(dataset, params, test_freq=100, verbose=False):
	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	v = tf.Variable(tf.truncated_normal([params.n], stddev=0.01, dtype=tf.float64))
	G = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.n, params.r], stddev=0.01, dtype=tf.float64))

	W1, f_A, f_B, v_A, v_B = circ_sparsity_recon(G, H, params.n, params.r, params.learn_corner, 
		params.n_diag_learned, params.init_type, params.init_stddev)

	y = compute_y(x, W1, params)
	
	y_ = tf.placeholder(tf.float64, [None, params.out_size])
	
	loss, accuracy = compute_loss_and_accuracy(y, y_, params)
	
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
