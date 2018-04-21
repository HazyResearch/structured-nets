import tensorflow as tf
import io,os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../tensorflow/')
from utils import *
from PIL import Image
import numpy as np
import time

# One image with params.num_pred_plot: image, caption - actual and predicted
# Here assuming classification
# x: (n, input dimension)
# y: (n)
# pred: (n)
# Each set of plots is for one image only
# Each row (for specific value of i): 
# orig image (caption: actual/predicted class),B^ix, GH^T(B^ix),A^i(GH^T(B^ix))
# num rows: len(viz_powers)
# ncols: 4

ncols = 3
ram = io.StringIO()

def show_learned_operators(vis_path,A,B,W,step):	
	"""
	print('A: ', A.shape)
	print('B: ', B.shape)
	print('W: ', W.shape)
	"""
	plt.clf()
	f, plots = plt.subplots(3,figsize=(5,15))
	plots[0].imshow(A)
	plots[0].set_title('A')
	plots[1].imshow(B)
	plots[1].set_title('B')
	plots[2].imshow(W)
	plots[2].set_title('W')
	plots[0].axis('off')
	plots[1].axis('off')
	plots[2].axis('off')
	plt.savefig(os.path.join(vis_path, str(step) + '_A_B_W.png'))
	plt.close()

# Assume all have been reshaped to be images
def show_prediction(vis_path,idx,viz_powers,image,true,pred,Bis,GHTBis,AiGHTBis,step):
	plt.clf()
	f, plots = plt.subplots(len(viz_powers)+1,ncols,figsize=(20,20))

	for row in range(len(viz_powers)+1):
		for col in range(ncols):
			plots[row, col].axis('off')	

	plots[0, 1].imshow(image)
	caption = 'Orig. Im., True: ' + str(true) + '; Pred: ' + str(pred)
	if true == pred:
		plots[0, 1].set_title(caption, color='green')
	else:
		plots[0, 1].set_title(caption, color='red')
	
	for row in range(len(viz_powers)):	
		Bi = Bis[row][idx,:].reshape((image.shape[0],image.shape[1]))
		GHTBi = GHTBis[row][idx,:].reshape((image.shape[0],image.shape[1]))
		AiGHTBi = AiGHTBis[row][idx,:].reshape((image.shape[0],image.shape[1]))

		plots[row+1,0].imshow(Bi)
		plots[row+1,0].set_title(r'$B^{' + str(viz_powers[row]) + '}x$', color='green')
		plots[row+1,1].imshow(GHTBi)
		plots[row+1,1].set_title(r'$GH^TB^{' + str(viz_powers[row]) + '}x$', color='green')
		plots[row+1,2].imshow(AiGHTBi)
		plots[row+1,2].set_title(r'$A^{' + str(viz_powers[row]) + '}GH^TB^{' + str(viz_powers[row]) + '}x$', color='green')

	plt.savefig(os.path.join(vis_path, str(step) + '_predictions_' + str(idx) + '.png'))
	"""
	plt.savefig(ram,format='png')
	ram.seek(0)
	im = Image.open(ram)
	im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
	im2.save('predictions' + str(idx) + '.png', format='PNG')
	"""
	plt.close()

def show_predictions(vis_path,step,num_pred_plot,layer_size,viz_powers,x,y,pred,Bis,GHTBis,AiGHTBis):
	assert num_pred_plot == x.shape[0] == y.size == pred.size
	img_size = np.sqrt(layer_size)
	assert img_size.is_integer
	img_size = int(img_size)
	nrows = len(viz_powers)

	f, plots = plt.subplots(num_pred_plot,ncols,figsize=(20,20))
	times = 0

	for idx in range(num_pred_plot):
		this_image = x[idx].reshape((img_size, img_size))

		# Get correct
		this_true = y[idx]

		# Get predicted
		this_pred = pred[idx]

		t1 = time.time()
		show_prediction(vis_path,idx,viz_powers,this_image,this_true,
			this_pred,Bis,GHTBis,AiGHTBis,step)
		times += (time.time() - t1)
	print('Average time of show_prediction: ', times/num_pred_plot)

def visualize_predictions(params,x,y,pred):
	return show_predictions(params.num_pred_plot,params.layer_size,x,y,pred)

def compute_powers(powers,A,GHT,B,x):
	Bis = []
	GHTBis = []
	AiGHTBis = []

	for power in powers:
		A_i = np.linalg.matrix_power(A,power)
		B_i = np.linalg.matrix_power(B,power)
		#print('B_i: ', B_i)
		GHTB_i = np.dot(GHT, B_i)
		A_iGHTB_i = np.dot(A_i, GHTB_i)
		Bis.append(np.dot(B_i, x.T).T)

		#print('x: ', x)
		#print('B_ix: ', Bis[-1])

		GHTBis.append(np.dot(GHTB_i, x.T).T)
		AiGHTBis.append(np.dot(A_iGHTB_i, x.T).T)

	return Bis,GHTBis,AiGHTBis 

def make_plots_params(params,A,B,G,H,W,x,y,pred,step):
	"""
	print('A: ', A.shape)
	print('B: ', B.shape)
	print('W: ', W.shape)
	"""
	make_plots(params.vis_path,params.num_pred_plot,params.layer_size,params.viz_powers,A,B,G,H,W,x,y,pred,step)

	# Just A,B,W
	show_learned_operators(params.vis_path,A,B,W,step)

def make_plots(vis_path, num_pred_plot,layer_size,viz_powers,A,B,G,H,W,x,y,pred,step):
	"""
	print('x.shape: ', x.shape)
	print('y.shape: ', y.shape)
	print('pred.shape: ', pred.shape)
	"""
	assert x.shape[0] == y.size == pred.size
	idx = np.random.randint(x.shape[0], size=num_pred_plot)
	x = x[idx,:]
	y = y[idx]
	pred = pred[idx]
	assert x.shape[0] == y.size == pred.size


	# GH^Tx
	low_rank = np.dot(G,H.T)
	low_rank_pred = np.dot(low_rank,x.T).T

	# B^ix, various i
	# GH^T(B^ix), various i
	# A^i(GH^T(B^ix)), various i
	t1 = time.time()
	Bis,GHTBis,AiGHTBis = compute_powers(viz_powers,A,low_rank,B,x)
	print('Time of compute_powers: ', time.time() - t1)


	# Various inputs, predictions, and ground truth
	show_predictions(vis_path,step,num_pred_plot,layer_size,viz_powers,x,y,pred,Bis,GHTBis,AiGHTBis)
	
def get_model_params(params,x,y_,batch_xs,batch_ys,sess,model):
	G,H = sess.run([model['G'], model['H']], feed_dict={x: batch_xs, y_: batch_ys})
	W = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
	if params.class_type == 'circulant_sparsity':
		# Construct A
		f_x_A = sess.run(model['f_x_A'], feed_dict={x: batch_xs, y_: batch_ys})
		# Construct B
		f_x_B = sess.run(model['f_x_B'], feed_dict={x: batch_xs, y_: batch_ys})

		if params.fix_A_identity:
			A = np.eye(params.layer_size)
		else:
			A = gen_Z_f(params.layer_size, f_x_B[0], f_x_B[1:]).T

		B = gen_Z_f(params.layer_size, f_x_B[0], f_x_B[1:])
	elif params.class_type == 'tridiagonal_corner':
		# Construct A
		supdiag_A = sess.run(model['supdiag_A'], feed_dict={x: batch_xs, y_: batch_ys})
		diag_A = sess.run(model['diag_A'], feed_dict={x: batch_xs, y_: batch_ys})
		subdiag_A = sess.run(model['subdiag_A'], feed_dict={x: batch_xs, y_: batch_ys})
		f_A = sess.run(model['f_A'], feed_dict={x: batch_xs, y_: batch_ys})
		# Construct B
		supdiag_B = sess.run(model['supdiag_B'], feed_dict={x: batch_xs, y_: batch_ys})
		diag_B = sess.run(model['diag_B'], feed_dict={x: batch_xs, y_: batch_ys})
		subdiag_B = sess.run(model['subdiag_B'], feed_dict={x: batch_xs, y_: batch_ys})
		f_B = sess.run(model['f_B'], feed_dict={x: batch_xs, y_: batch_ys})

		# Check if this is transpose
		A = gen_tridiag_corner(subdiag_A, supdiag_A, diag_A, f_A)
		B = gen_tridiag_corner(subdiag_B, supdiag_B, diag_B, f_B)
	else:
		print('Class type not supported: ', params.class_type)
		assert 0
	"""
	print('A: ', A.shape)
	print('B: ', B.shape)
	print('W: ', W.shape)
	"""
	return A,B,G,H,W

def visualize(params,sess,model,x,y_,batch_xs,batch_ys,y_pred,this_step):
	A,B,G,H,W = get_model_params(params,x,y_,batch_xs,batch_ys,sess,model)

	"""
	print('A: ', A.shape)
	print('B: ', B.shape)
	print('W: ', W.shape)
	print('A: ', A)
	print('B: ', B)
	print('G: ', G)
	print('H: ', H)
	quit()
	"""
	y_true = np.argmax(batch_ys,axis=1)
	y_pred = np.argmax(y_pred,axis=1)
	make_plots_params(params,A,B,G,H,W,batch_xs,y_true,y_pred,this_step)

if __name__ == '__main__':
	num_pred_plot = 5
	img_size = 2
	layer_size = img_size**2
	r = 1
	A = gen_Z_f(layer_size,1)#np.random.random((layer_size,layer_size))
	B = gen_Z_f(layer_size,-1)#np.random.random((layer_size,layer_size))
	G = np.random.random((layer_size,r))
	H = np.random.random((layer_size,r))
	n = 100
	x = np.random.random((n,layer_size))
	y = np.random.randint(low=0, high=10,size=n)
	pred = np.random.randint(low=0,high=10,size=n)
	viz_powers = [1,5,10]
	make_plots(num_pred_plot,layer_size,viz_powers,A,B,G,H,x,y,pred)
