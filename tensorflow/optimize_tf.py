import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import pickle as pkl
from utils import *
from reconstruction import *
from visualize import visualize
from model import *
import time
import logging

def restore_from_checkpoint(dataset, params, sess, saver, x, y_, loss, accuracy):
    # Restore the best validation checkpoint, test on that
    saver.restore(sess, tf.train.latest_checkpoint(params.checkpoint_path))
    print('Restored from most recent checkpoint: ')
    val_loss, val_accuracy = sess.run([loss, accuracy], feed_dict={x: dataset.val_X, y_: dataset.val_Y})
    print('After restoring, val loss and accuracy: %f, %f' % (val_loss, val_accuracy))
    return val_loss, val_accuracy

def optimize_tf(dataset, params):
    # Create model
    x = tf.placeholder(tf.float64, [None, params.input_size],name='x')
    y, model = forward(x, params)
    y_ = tf.placeholder(tf.float64, [None, params.out_size],name='y_')
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
    if params.restore_from_checkpoint:
        val_loss,val_accuracy = restore_from_checkpoint(dataset,params,sess,saver,x,y_,loss,accuracy)

    eigvals = {'E': [], 'W': [], 'A': [], 'B': []}
    model_params = {'E': [], 'W': [], 'A': [], 'B': []}
    losses = {'train': [], 'val': [], 'DR': [], 'ratio': [], 'eigvals': eigvals, 'params': model_params}
    accuracies = {'train': [], 'val': [], 'best_val': 0.0, 'best_val_iter': 0}
    t1 = time.time()
    for _ in range(params.steps):
        this_step, lr = sess.run([global_step, learning_rate])
        batch_xs, batch_ys = dataset.batch(params.batch_size, this_step)
        _ = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

        if this_step % params.test_freq == 0:
            logging.debug(time.time() - t1)
            t1 = time.time()
            logging.debug('Training step: ' + str(this_step))
            # Verify displacement rank
            if params.check_disp and this_step % params.check_disp_freq == 0:
                dr, norm_res, norm_W, E_ev, W_ev, A_ev, B_ev = check_rank(sess, x, y_, batch_xs, batch_ys, params, model)
                losses['DR'].append(dr)
                losses['norm_res'].append(norm_res)
                losses['norm_W'].append(norm_W)
                losses['eigvals']['E'].append(E_ev)
                losses['eigvals']['W'].append(W_ev)
                losses['eigvals']['A'].append(A_ev)
                losses['eigvals']['B'].append(B_ev)
            train_loss, train_accuracy, train_loss_summ, train_acc_summ, y_pred = sess.run([loss, accuracy, train_loss_summary,
                train_acc_summary, y], feed_dict={x: batch_xs, y_: batch_ys})
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

            # Save
            pkl.dump(losses, open(params.result_path + '_losses.p', 'wb'), protocol=2)
            pkl.dump(accuracies, open(params.result_path + '_accuracies.p', 'wb'), protocol=2)

            logging.debug('Saved losses, accuracies to: %s' % (params.result_path))
            logging.debug('Train loss, accuracy for class %s: %f, %f' % (params.class_type, train_loss, train_accuracy))
            logging.debug('Validation loss, accuracy %s: %f, %f' % (params.class_type, val_loss, val_accuracy))
            logging.debug("Best validation accuracy so far: %f" % accuracies['best_val'])

        # Update checkpoint if better validation accuracy
        if val_accuracy > accuracies['best_val']:
            accuracies['best_val'] = val_accuracy
            accuracies['best_val_iter'] = this_step
            #if this_step > 0 and this_step % params.checkpoint_freq == 0:
            #save_path = saver.save(sess, os.path.join(params.checkpoint_path, str(this_step)))
            save_path = saver.save(sess, os.path.join(params.checkpoint_path, str(this_step) + '_' + str(accuracies['best_val'])))
            logging.debug("Updating validation accuracy so far: %f" % accuracies['best_val'])
            logging.debug("Model saved in file: %s" % save_path)


        elif accuracies['best_val_iter'] <= this_step - params.early_stop_steps:
            logging.debug('Early stopping: best val iter at %d, current step %d' %(accuracies['best_val_iter'], this_step))
            break

        if this_step > 0 and params.viz_freq > 0 and this_step % params.viz_freq == 0:
            visualize(params,sess,model,x,y_,batch_xs,batch_ys,y_pred,this_step)

    # Get final params
    if params.check_disp:
        dr, norm_res, norm_W, E_ev, W_ev, A_ev, B_ev, E, W, A, B = check_rank(sess, x, y_, batch_xs, batch_ys, params, model)
        losses['DR'].append(dr)
        losses['norm_res'].append(norm_res)
        losses['norm_W'].append(norm_W)
        losses['params']['E'] = E
        losses['params']['W'] = W
        losses['params']['A'] = A
        losses['params']['B'] = B

    # Test trained model
    if params.test:
        # Load test
        dataset.load_test_data()
        # Test on the current model
        if not params.test_best_val_checkpoint:
            test_loss, test_accuracy, test_loss_summ, test_acc_summ = sess.run([loss, accuracy, test_loss_summary, test_acc_summary], feed_dict={x: dataset.test_X, y_: dataset.test_Y})

            summary_writer.add_summary(test_loss_summ, this_step)
            summary_writer.add_summary(test_acc_summ, this_step)

            logging.debug('Test loss, %s: %f' % (params.class_type, test_loss))
            logging.debug('Test accuracy, %s: %f ' % (params.class_type, test_accuracy))

            losses['test'] = test_loss
            accuracies['test'] = test_accuracy

        else:
            restore_from_checkpoint(dataset, params, sess, saver, x, y_, loss, accuracy)
            test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={x: dataset.test_X, y_: dataset.test_Y})
            
            logging.debug('Test loss of best val checkpoint, %s: %f' % (params.class_type, test_loss))
            logging.debug('Test accuracy of best val checkpoint, %s: %f ' % (params.class_type, test_accuracy))

            losses['test_best_val'] = test_loss
            accuracies['test_best_val'] = test_accuracy

    return losses, accuracies
