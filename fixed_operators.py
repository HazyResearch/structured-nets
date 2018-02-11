import numpy as np
import os
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
	
	#train_loss_summary = tf.summary.scalar('train_loss', loss)
	#train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
	val_loss_summary = tf.summary.scalar('val_loss', loss)
	val_acc_summary = tf.summary.scalar('val_accuracy', accuracy)
	test_loss_summary = tf.summary.scalar('test_loss', loss)
	test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)

	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()

	step = 0

	losses = {}
	accuracies = {}
	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size, step)
		_ = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
	 
		if step % test_freq == 0:
			print('Training step: ', step)
			# Verify displacement rank
			if params.check_disp:
				v_real, W1_real = sess.run([v, W1], feed_dict={x: batch_xs, y_: batch_ys})
				A = np.diag(v_real)
				E = W1_real - np.dot(A, np.dot(W1_real, B_vand))
				print('Disp rank: ', np.linalg.matrix_rank(E))

			#train_loss, train_accuracy, train_loss_summ, train_acc_summ = sess.run([loss, accuracy, train_loss_summary, 
			#	train_acc_summary], feed_dict={x: batch_xs, y_: batch_ys})
			val_loss, val_accuracy, val_loss_summ, val_acc_summ = sess.run([loss, accuracy, val_loss_summary, 
				val_acc_summary], feed_dict={x: dataset.val_X, y_: dataset.val_Y})			
			
			#summary_writer.add_summary(train_loss_summ, step)
			#summary_writer.add_summary(train_acc_summ, step)
			summary_writer.add_summary(val_loss_summ, step)
			summary_writer.add_summary(val_acc_summ, step)

			#train_losses.append(train_loss)
			#train_accuracies.append(train_accuracy)
			val_losses.append(val_loss)
			val_accuracies.append(val_accuracy)
			
			#print('Train loss, accuracy: ', train_loss, train_accuracy)
			print('Validation loss, accuracy: ', val_loss, val_accuracy)

			if verbose:
				print('Current W1: ', W1_real)

		if step % params.checkpoint_freq == 0:
			save_path = saver.save(sess, os.path.join(params.checkpoint_path, str(step)))
			print("Model saved in file: %s" % save_path)

		step += 1

	losses['train'] = train_losses
	losses['val'] = val_losses
	accuracies['train'] = train_accuracies
	accuracies['val'] = val_accuracies

	# Test trained model
	if params.test:
		# Load test
		dataset.load_test_data()
		test_loss, test_accuracy, test_loss_summ, test_acc_summ = sess.run([loss, accuracy, test_loss_summary, test_acc_summary], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

		summary_writer.add_summary(test_loss_summ, step)
		summary_writer.add_summary(test_acc_summ, step)

		print('SGD test loss, Vandermonde-like: ', test_loss)
		print('SGD test accuracy, Vandermonde-like: ', test_accuracy)

		losses['test'] = test_loss
		accuracies['test'] = test_accuracy

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
	
	#train_loss_summary = tf.summary.scalar('train_loss', loss)
	#train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
	val_loss_summary = tf.summary.scalar('val_loss', loss)
	val_acc_summary = tf.summary.scalar('val_accuracy', accuracy)
	test_loss_summary = tf.summary.scalar('test_loss', loss)
	test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)

	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()

	step = 0

	losses = {}
	accuracies = {}
	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size, step)
		_ = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})
 
		if step % test_freq == 0:
			print('Training step: ', step)
			if params.check_disp:
				# Verify displacement rank
				W1_real = sess.run(W1, feed_dict={x: batch_xs, y_: batch_ys})
				E = W1_real - np.dot(A, np.dot(W1_real, B))
				print('Disp rank: ', np.linalg.matrix_rank(E))
			
			#train_loss, train_accuracy, train_loss_summ, train_acc_summ = sess.run([loss, accuracy, train_loss_summary, 
			#	train_acc_summary], feed_dict={x: batch_xs, y_: batch_ys})
			val_loss, val_accuracy, val_loss_summ, val_acc_summ = sess.run([loss, accuracy, val_loss_summary, 
				val_acc_summary], feed_dict={x: dataset.val_X, y_: dataset.val_Y})	

			#summary_writer.add_summary(train_loss_summ, step)
			#summary_writer.add_summary(train_acc_summ, step)
			summary_writer.add_summary(val_loss_summ, step)
			summary_writer.add_summary(val_acc_summ, step)

			#train_losses.append(train_loss)
			#train_accuracies.append(train_accuracy)
			val_losses.append(val_loss)
			val_accuracies.append(val_accuracy)
			
			#print('Train loss, accuracy: ', train_loss, train_accuracy)
			print('Validation loss, accuracy: ', val_loss, val_accuracy)

			if verbose:
				print('Current W1: ', W1_real)

		if step % params.checkpoint_freq == 0:
			save_path = saver.save(sess, os.path.join(params.checkpoint_path, str(step)))
			print("Model saved in file: %s" % save_path)

		step += 1

	losses['train'] = train_losses
	losses['val'] = val_losses
	accuracies['train'] = train_accuracies
	accuracies['val'] = val_accuracies

	# Test trained model
	if params.test:
		# Load test
		dataset.load_test_data()

		test_loss, test_accuracy, test_loss_summ, test_acc_summ = sess.run([loss, accuracy, test_loss_summary, test_acc_summary], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

		summary_writer.add_summary(test_loss_summ, step)
		summary_writer.add_summary(test_acc_summ, step)

		print('SGD test loss, Hankel-like: ', test_loss)
		print('SGD test accuracy, Hankel-like: ', test_accuracy)
		losses['test'] = test_loss
		accuracies['test'] = test_accuracy

	return losses, accuracies


def toeplitz_like(dataset, params, test_freq=100, verbose=False):
	A = gen_Z_f(params.layer_size, 1)
	B = gen_Z_f(params.layer_size, -1)

	# Create the model
	x = tf.placeholder(tf.float64, [None, params.input_size])
	G = tf.Variable(tf.truncated_normal([params.layer_size, params.r], stddev=0.01, dtype=tf.float64))
	H = tf.Variable(tf.truncated_normal([params.layer_size, params.r], stddev=0.01, dtype=tf.float64))

	W1 = toeplitz_like_recon(G, H, params.layer_size, params.r)

	y = compute_y(x, W1, params)
	
	y_ = tf.placeholder(tf.float64, [None, params.out_size])
	
	loss, accuracy = compute_loss_and_accuracy(y, y_, params)

	#train_loss_summary = tf.summary.scalar('train_loss', loss)
	#train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
	val_loss_summary = tf.summary.scalar('val_loss', loss)
	val_acc_summary = tf.summary.scalar('val_accuracy', accuracy)
	test_loss_summary = tf.summary.scalar('test_loss', loss)
	test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)


	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()

	step = 0

	losses = {}
	accuracies = {}
	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size, step)
		_, = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

	 
		if step % test_freq == 0:
			print('Training step: ', step)
			if params.check_disp:
				# Verify displacement rank
				W1_real = sess.run(W1, feed_dict={x: batch_xs, y_: batch_ys})
				E_sylv = np.dot(A, W1_real) - np.dot(W1_real, B)
				print('Disp rank, Sylv: ', np.linalg.matrix_rank(E_sylv))

			#train_loss, train_accuracy, train_loss_summ, train_acc_summ = sess.run([loss, accuracy, train_loss_summary, 
			#	train_acc_summary], feed_dict={x: batch_xs, y_: batch_ys})
			val_loss, val_accuracy, val_loss_summ, val_acc_summ = sess.run([loss, accuracy, val_loss_summary, 
				val_acc_summary], feed_dict={x: dataset.val_X, y_: dataset.val_Y})				
			
			#summary_writer.add_summary(train_loss_summ, step)
			#summary_writer.add_summary(train_acc_summ, step)
			summary_writer.add_summary(val_loss_summ, step)
			summary_writer.add_summary(val_acc_summ, step)

			#train_losses.append(train_loss)
			#train_accuracies.append(train_accuracy)
			val_losses.append(val_loss)
			val_accuracies.append(val_accuracy)
			
			#print('Train loss, accuracy: ', train_loss, train_accuracy)
			print('Validation loss, accuracy: ', val_loss, val_accuracy)

			if verbose:
				print('Current W1: ', W1_real)

		if step % params.checkpoint_freq == 0:
			save_path = saver.save(sess, os.path.join(params.checkpoint_path, str(step)))
			print("Model saved in file: %s" % save_path)

		step += 1

	losses['train'] = train_losses
	losses['val'] = val_losses
	accuracies['train'] = train_accuracies
	accuracies['val'] = val_accuracies

	# Test trained model
	if params.test:
		# Load test
		dataset.load_test_data()

		test_loss, test_accuracy, test_loss_summ, test_acc_summ = sess.run([loss, accuracy, test_loss_summary, 
			test_acc_summary], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

		summary_writer.add_summary(test_loss_summ, step)
		summary_writer.add_summary(test_acc_summ, step)

		print('SGD test loss, Toeplitz-like: ', test_loss)
		print('SGD test accuracy, Toeplitz-like: ', test_accuracy)
		losses['test'] = test_loss
		accuracies['test'] = test_accuracy

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
	#train_loss_summary = tf.summary.scalar('train_loss', loss)
	#train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
	val_loss_summary = tf.summary.scalar('val_loss', loss)
	val_acc_summary = tf.summary.scalar('val_accuracy', accuracy)
	test_loss_summary = tf.summary.scalar('test_loss', loss)
	test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)


	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()

	step = 0

	losses = {}
	accuracies = {}
	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size, step)
		_, = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

	 
		if step % test_freq == 0:
			print('Training step: ', step)
			#train_loss, train_accuracy, train_loss_summ, train_acc_summ = sess.run([loss, accuracy, train_loss_summary, 
			#	train_acc_summary], feed_dict={x: batch_xs, y_: batch_ys})
			val_loss, val_accuracy, val_loss_summ, val_acc_summ = sess.run([loss, accuracy, val_loss_summary, 
				val_acc_summary], feed_dict={x: dataset.val_X, y_: dataset.val_Y})				
			
			#summary_writer.add_summary(train_loss_summ, step)
			#summary_writer.add_summary(train_acc_summ, step)
			summary_writer.add_summary(val_loss_summ, step)
			summary_writer.add_summary(val_acc_summ, step)

			#train_losses.append(train_loss)
			#train_accuracies.append(train_accuracy)
			val_losses.append(val_loss)
			val_accuracies.append(val_accuracy)
			
			#print('Train loss, accuracy: ', train_loss, train_accuracy)
			print('Validation loss, accuracy: ', val_loss, val_accuracy)

			if verbose:
				print('Current W1: ', W1_real)

		if step % params.checkpoint_freq == 0:
			save_path = saver.save(sess, os.path.join(params.checkpoint_path, str(step)))
			print("Model saved in file: %s" % save_path)

		step += 1

	losses['train'] = train_losses
	losses['val'] = val_losses
	accuracies['train'] = train_accuracies
	accuracies['val'] = val_accuracies

	# Test trained model
	if params.test:
		# Load test
		dataset.load_test_data()

		test_loss, test_accuracy, test_loss_summ, test_acc_summ = sess.run([loss, accuracy, test_loss_summary, test_acc_summary], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

		summary_writer.add_summary(test_loss_summ, step)
		summary_writer.add_summary(test_acc_summ, step)
		
		print('SGD test loss, low rank: ', test_loss)
		print('SGD test accuracy, low rank: ', test_accuracy)
		losses['test'] = test_loss
		accuracies['test'] = test_accuracy

	return losses, accuracies

def unconstrained(dataset, params, test_freq=100, verbose=False):
	# Create the model
	x = tf.placeholder(tf.float64, [None, params.n])
	W1 = tf.Variable(tf.truncated_normal([params.n, params.n], stddev=0.01, dtype=tf.float64))
	y = compute_y(x, W1, params)
	y_ = tf.placeholder(tf.float64, [None, params.out_size])
	
	loss, accuracy = compute_loss_and_accuracy(y, y_, params)
	
	#train_loss_summary = tf.summary.scalar('train_loss', loss)
	#train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
	val_loss_summary = tf.summary.scalar('val_loss', loss)
	val_acc_summary = tf.summary.scalar('val_accuracy', accuracy)
	test_loss_summary = tf.summary.scalar('test_loss', loss)
	test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)


	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	train_step = tf.train.MomentumOptimizer(params.lr, params.mom).minimize(loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()

	step = 0

	losses = {}
	accuracies = {}
	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	while step < params.steps:
		batch_xs, batch_ys = dataset.batch(params.batch_size, step)
		_ = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

	 
		if step % test_freq == 0:
			print('Training step: ', step)
			#train_loss, train_accuracy, train_loss_summ, train_acc_summ = sess.run([loss, accuracy, train_loss_summary, 
			#	train_acc_summary], feed_dict={x: batch_xs, y_: batch_ys})
			val_loss, val_accuracy, val_loss_summ, val_acc_summ = sess.run([loss, accuracy, val_loss_summary, 
				val_acc_summary], feed_dict={x: dataset.val_X, y_: dataset.val_Y})				
			
			#summary_writer.add_summary(train_loss_summ, step)
			#summary_writer.add_summary(train_acc_summ, step)
			summary_writer.add_summary(val_loss_summ, step)
			summary_writer.add_summary(val_acc_summ, step)

			#train_losses.append(train_loss)
			#train_accuracies.append(train_accuracy)
			val_losses.append(val_loss)
			val_accuracies.append(val_accuracy)
			
			#print('Train loss, accuracy: ', train_loss, train_accuracy)
			print('Validation loss, accuracy: ', val_loss, val_accuracy)

			if verbose:
				print('Current W1: ', W1_real)

		if step % params.checkpoint_freq == 0:
			save_path = saver.save(sess, os.path.join(params.checkpoint_path, str(step)))
			print("Model saved in file: %s" % save_path)

		step += 1

	losses['train'] = train_losses
	losses['val'] = val_losses
	accuracies['train'] = train_accuracies
	accuracies['val'] = val_accuracies

	# Test trained model
	if params.test:
		# Load test
		dataset.load_test_data()

		test_loss, test_accuracy, test_loss_summ, test_acc_summ = sess.run([loss, accuracy, test_loss_summary, 
			test_acc_summary], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

		summary_writer.add_summary(test_loss_summ, step)
		summary_writer.add_summary(test_acc_summ, step)
		
		print('SGD test loss, unconstrained: ', test_loss)
		print('SGD test accuracy, unconstrained: ', test_accuracy)
		losses['test'] = test_loss
		accuracies['test'] = test_accuracy

	return losses, accuracies

