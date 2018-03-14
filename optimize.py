import numpy as np
import os
import tensorflow as tf
from utils import *
from reconstruction import *
from model import *

def optimize(dataset, params):
	# Create model
	x = tf.placeholder(tf.float64, [None, params.input_size])
	y, model = forward(x, params)
	y_ = tf.placeholder(tf.float64, [None, params.out_size])
	loss, accuracy = compute_loss_and_accuracy(y, y_, params)
	
	train_loss_summary = tf.summary.scalar('train_loss', loss)
	train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
	val_loss_summary = tf.summary.scalar('val_loss', loss)
	val_acc_summary = tf.summary.scalar('val_accuracy', accuracy)
	test_loss_summary = tf.summary.scalar('test_loss', loss)
	test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)

	summary_writer = tf.summary.FileWriter(params.log_path, graph=tf.get_default_graph())

	# Allow for decay of learning rate
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(params.lr, global_step,
                                           int(params.decay_freq*params.steps), params.decay_rate, staircase=True)	

	train_step = tf.train.MomentumOptimizer(learning_rate, params.mom).minimize(loss, global_step=global_step)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()

	losses = {'train': [], 'val': [], 'DR': [], 'ratio': []}
	accuracies = {'train': [], 'val': []}

	for _ in range(params.steps):
		this_step, lr = sess.run([global_step, learning_rate])
		batch_xs, batch_ys = dataset.batch(params.batch_size, this_step)
		_ = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

		if this_step % params.test_freq == 0:
			print('Training step: ', this_step)
			# Verify displacement rank
			if params.check_disp:
				dr, ratio = check_rank(sess, x, y_, batch_xs, batch_ys, params, model)
				losses['DR'].append(dr)
				losses['ratio'].append(ratio)

			train_loss, train_accuracy, train_loss_summ, train_acc_summ = sess.run([loss, accuracy, train_loss_summary, 
				train_acc_summary], feed_dict={x: batch_xs, y_: batch_ys})
			val_loss, val_accuracy, val_loss_summ, val_acc_summ = sess.run([loss, accuracy, val_loss_summary, 
				val_acc_summary], feed_dict={x: dataset.val_X, y_: dataset.val_Y})			
			
			summary_writer.add_summary(train_loss_summ, this_step)
			summary_writer.add_summary(train_acc_summ, this_step)
			summary_writer.add_summary(val_loss_summ, this_step)
			summary_writer.add_summary(val_acc_summ, this_step)

			losses['train'].append(train_loss)
			accuracies['train'].append(train_accuracy)
			losses['val'].append(val_loss)
			accuracies['val'].append(val_accuracy)
			
			print('Train loss, accuracy for class %s: %f, %f' % (params.class_type, train_loss, train_accuracy))
			print('Validation loss, accuracy %s: %f, %f' % (params.class_type, val_loss, val_accuracy))

		if this_step % params.checkpoint_freq == 0:
			save_path = saver.save(sess, os.path.join(params.checkpoint_path, str(this_step)))
			print("Model saved in file: %s" % save_path)

	# Test trained model
	if params.test:
		# Load test
		dataset.load_test_data()
		test_loss, test_accuracy, test_loss_summ, test_acc_summ = sess.run([loss, accuracy, test_loss_summary, test_acc_summary], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

		summary_writer.add_summary(test_loss_summ, this_step)
		summary_writer.add_summary(test_acc_summ, this_step)

		print('Test loss, %s: %f' % (params.class_type, test_loss))
		print('Test accuracy, %s: %f ' % (params.class_type, test_accuracy))

		losses['test'] = test_loss
		accuracies['test'] = test_accuracy

	return losses, accuracies



