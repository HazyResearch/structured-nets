import numpy as np
import os, sys
sys.path.insert(0, '../../pytorch/')
from nets import construct_net
import time
import torch
from torch_utils import *
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

def optimize_torch(dataset, params):
	writer = SummaryWriter(params.log_path)
	net = construct_net(params)

	net.cuda()

	print(torch.cuda.get_device_name(0))

	for name, param in net.named_parameters():
	    if param.requires_grad:
	        print(name) #, param.data

	optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=params.mom)

	losses = {'train': [], 'val': [], 'DR': [], 'ratio': []}
	accuracies = {'train': [], 'val': []}

	val_X, val_Y = Variable(torch.FloatTensor(dataset.val_X).cuda()), Variable(torch.FloatTensor(dataset.val_Y).cuda())


	loss_fn = get_loss(params)

	t1 = time.time()
	for step in range(params.steps):
		batch_xs, batch_ys = dataset.batch(params.batch_size, step)
		batch_xs, batch_ys = Variable(torch.FloatTensor(batch_xs).cuda()), Variable(torch.FloatTensor(batch_ys).cuda())
		optimizer.zero_grad()   # zero the gradient buffers

		output = net.forward(batch_xs)


		train_loss, train_accuracy = compute_loss_and_accuracy(output, batch_ys, params, loss_fn)

		train_loss.backward()

		optimizer.step()

		if step % params.test_freq == 0:
			print('Time: ', time.time() - t1)
			t1 = time.time()
			losses['train'].append(train_loss)
			accuracies['train'].append(train_accuracy)
			writer.add_scalar('Train/Loss', train_loss, step)
			writer.add_scalar('Train/Accuracy', train_accuracy, step)

			print(('Training step: ', step))

			# Test on validation set
			output = net.forward(val_X)
			val_loss, val_accuracy = compute_loss_and_accuracy(output, val_Y, params, loss_fn)

			writer.add_scalar('Val/Loss', val_loss, step)
			writer.add_scalar('Val/Accuracy', val_accuracy, step)

			losses['val'].append(val_loss)
			accuracies['val'].append(val_accuracy)
			
			print(('Train loss, accuracy for class %s: %f, %f' % (params.class_type, train_loss, train_accuracy)))
			print(('Validation loss, accuracy %s: %f, %f' % (params.class_type, val_loss, val_accuracy)))

		if step % params.checkpoint_freq == 0:
			save_path = os.path.join(params.checkpoint_path, str(step))
			with open(save_path, 'wb') as f: 
				torch.save(net.state_dict(), f)
			print(("Model saved in file: %s" % save_path))

	# Test trained model
	if params.test:
		# Load test
		dataset.load_test_data()
		test_X, test_Y = Variable(torch.FloatTensor(dataset.test_X).cuda()), Variable(torch.FloatTensor(dataset.test_Y).cuda())

		output = net.forward(test_X)
		test_loss, test_accuracy = compute_loss_and_accuracy(output, test_Y, params, loss_fn)

		writer.add_scalar('Test/Loss', test_loss)
		writer.add_scalar('Test/Accuracy', test_accuracy)

		print(('Test loss, %s: %f' % (params.class_type, test_loss)))
		print(('Test accuracy, %s: %f ' % (params.class_type, test_accuracy)))

		losses['test'] = test_loss
		accuracies['test'] = test_accuracy


	writer.export_scalars_to_json(os.path.join(params.log_path, "all_scalars.json"))
	writer.close()

	return losses, accuracies
