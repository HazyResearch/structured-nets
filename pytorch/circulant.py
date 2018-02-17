import numpy as np
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from torch import autograd
from torch.autograd import Variable

from utils import get_train_valid_datasets, train, train_all_epochs, accuracy, all_losses

use_cuda = torch.cuda.is_available()

class TwoLayerNet(nn.Module):

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_features)
        self.fc2 = nn.Linear(n_features, n_classes)

    def forward(self, x):
        feat = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(feat)

    @staticmethod
    def loss(output, target, reduce=True):
        return F.cross_entropy(output, target, reduce=reduce)

    @staticmethod
    def predict(output):
        return output.data.max(1)[1]

class Circulant(nn.Module):
    def __init__(self, in_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, Krylov(shift, self.weight).t(), self.bias)


class TwoLayerCirculant(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc1 = Circulant(n_features, bias=True)
        self.fc2 = nn.Linear(n_features, n_classes)

    def forward(self, x):
        feat = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(feat)

    @staticmethod
    def loss(output, target, reduce=True):
        return F.cross_entropy(output, target, reduce=reduce)

    @staticmethod
    def predict(output):
        return output.data.max(1)[1]

def shift(v, f=1):
    return torch.cat((f * v[[v.size(0) - 1]], v[:-1]))

def Krylov(linear_map, v, n=None):
    if n is None:
        n = v.size(0)
    cols = [v]
    for _ in range(n - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)

batch_size = 256
if use_cuda:
    loader_args = {'num_workers': 8, 'pin_memory': True}
else:
    loader_args = {'num_workers': 1, 'pin_memory': False}

def loader_from_dataset(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=True, **loader_args)

mnist_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])
mnist_train = datasets.MNIST(
    '../data', train=True, download=True, transform=mnist_normalize)
# Just for faster training on CPU
# mnist_train.train_data = mnist_train.train_data[:5000]
mnist_test = datasets.MNIST(
    'data', train=False, download=True, transform=mnist_normalize)
mnist_train, mnist_valid = get_train_valid_datasets(mnist_train)
train_loader = loader_from_dataset(mnist_train)
valid_loader = loader_from_dataset(mnist_valid)
test_loader = loader_from_dataset(mnist_test)

n_features = 28 * 28
n_classes = 10
gamma = 0.003  # Best gamma is around 0.001--0.003
n_components = 10000
sgd_n_epochs = 15

def sgd_opt_from_model(model, learning_rate=0.01, momentum=0.9, weight_decay=0.001):
    return optim.SGD((p for p in model.parameters() if p.requires_grad),
                     lr=learning_rate, momentum=momentum,
                     weight_decay=weight_decay)

# model = TwoLayerNet(n_features, n_classes)
model = TwoLayerCirculant(n_features, n_classes)
optimizer = sgd_opt_from_model(model)
train_loss, train_acc, valid_acc = train_all_epochs(train_loader, valid_loader, model, optimizer, sgd_n_epochs)
correct, total = accuracy(test_loader, model)
print(f'Test set: Accuracy: {correct}/{total} ({correct/total*100:.4f}%)\n')
