import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Circulant sparsity pattern
def gen_Z_f(m, f, v=None):
    if v is not None:
        assert v.size <= m-1
    I_m = np.eye(m-1, m-1)
    Z_f = np.hstack((I_m, np.zeros((m-1, 1))))
    Z_f = np.vstack((np.zeros((1, m)), Z_f))
    Z_f[0, -1] = f
    if v is not None:
        for i in range(v.size):
            Z_f[i+1, i] = v[i]

    return Z_f

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def get_loss(params, generator=None, model_opt=None):
	if params.dataset.startswith('true'):
		assert params.loss == 'mse'
		return nn.MSELoss()
	elif params.dataset.startswith('copy'):
		assert params.loss == 'label_smoothing'
		ls = LabelSmoothing(params.ls_size, params.ls_padding_idx, params.ls_smoothing)
		return SimpleLossCompute(generator, ls, model_opt)

	else:
		assert params.loss == 'cross_entropy'
		return nn.CrossEntropyLoss()


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

# y_: One hot. y: vector of predicted class probabilities.
def compute_loss_and_accuracy(pred, true, params, loss_fn, ntokens=None):
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

	elif params.loss == 'label_smoothing':
		loss = loss_fn(pred, true, ntokens)
		accuracy = torch.FloatTensor([0])
		return loss, accuracy

	else:
		print(('Not supported: ', params.loss))
		assert 0
