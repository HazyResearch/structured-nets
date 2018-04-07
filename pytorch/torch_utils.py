import torch
import torch.nn as nn

def get_loss(params):
	if params.loss == 'mse':
		return nn.MSELoss()
	elif params.loss == 'cross_entropy':
		return nn.CrossEntropyLoss()

# y_: One hot. y: vector of predicted class probabilities.
def compute_loss_and_accuracy(pred, true, params, loss_fn):
	if params.loss == 'mse':
		mse = loss_fn(pred, true)
		accuracy = torch.FloatTensor([0])

		return mse, accuracy
	elif params.loss == 'cross_entropy':
		_, true_argmax = torch.max(true, 1)
		cross_entropy = loss_fn(pred, true_argmax)

		_, pred_argmax = torch.max(pred, 1)
		correct_prediction = torch.eq(true_argmax, pred_argmax)
		accuracy = torch.mean(correct_prediction.float())

		return cross_entropy, accuracy
	else:
		print('Not supported: ', params.loss)
		assert 0
