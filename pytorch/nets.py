import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import copy
import sys
sys.path.insert(0, '../../krylov/')
sys.path.insert(0, '../../pytorch/attention/')

from torch_krylov import *
from torch_reconstruction import *
from attention import *
import toeplitz_gpu as toep
import krylov_multiply as subd
import LDR as ldr
#from krylov_multiply import *
# import structured_layer # this is already in attention
# TODO fix the 'import *'s

def construct_net(params):
    if params.model == 'LeNet':
        return LeNet(params)
    elif params.model == 'MLP':
        return MLP(params)
    elif params.model == 'LDR':
        return LDRNet(params)
    elif params.model == 'LDRfat':
        return LDR2(params)
    elif params.model == 'RNN':
        return RNN(params)
    elif params.model == 'Attention':
        return Attention(params)
    elif params.model == 'CNN':
        return CNN(params)
    else:
        print(('Model not supported: ', params.model))
        assert 0


class LeNet(nn.Module):
    def __init__(self, params):
        super(LeNet, self).__init__()
        # self.W1 = get_structured_W(params)
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = nn.Conv2d(3, 3, 5, padding=2)
        self.fc = nn.Linear(3*1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 3*1024)
        x = F.relu(self.fc(x))
        x = self.fc3(x)
        return x

    def loss(self):
        return 0

class LDR2(nn.Module):
    def __init__(self, params):
        super(LDR2, self).__init__()

        fc_size = 1024

        self.params = params

        self.W = StructuredLinear(params)
        self.fc = nn.Linear(params.layer_size, fc_size)
        self.fc3 = nn.Linear(fc_size, 10)

    def forward(self, x):
        x = self.W(x)
        x = F.relu(self.fc(x))
        x = self.fc3(x)
        return x

    def loss(self):
        lamb = 0.0001
        if self.params.class_type == 'unconstrained':
            return 0 #TODO should probably compare against L2 regularized
        return lamb*torch.sum(torch.abs(self.W.G)) + lamb*torch.sum(torch.abs(self.W.H))



class LDRNet(nn.Module):
    def __init__(self, params):
        super(LDRNet, self).__init__()
        self.n = 1024
        channels = 3
        fc_size = 1024

        self.LDR1 = ldr.LDR(params.class_type, 3, 3, params.r, self.n, bias=True)
        # self.LDR2 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.LDR3 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.LDR4 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.b1 = Parameter(torch.Tensor(params.layer_size))
        # torch.nn.init.normal_(self.b1,std=params.init_stddev)
        # self.W1 = Parameter(torch.Tensor(3*1024, fc_size))
        self.fc = nn.Linear(3*self.n, fc_size)
        self.fc3 = nn.Linear(fc_size, 10)

    def forward(self, x):
        # print("\nx shape", x.shape)
        x = x.view(-1, 3, 1024)
        x = x.transpose(0,1).contiguous().view(3, -1, self.n)
        # print("x shape", x.shape)
        x = F.relu(self.LDR1(x))
        # x += self.b1
        # x = F.relu(self.LDR2(x))
        # x = F.relu(self.LDR3(x))
        # x = F.relu(self.LDR4(x))
        # x = x.view(-1, self.n)
        # print("x shape", x.shape)
        x = x.transpose(0,1) # swap batches and channels axis
        # print("x shape", x.shape)
        x = x.contiguous().view(-1, 3*self.n)
        # print("x shape", x.shape)
        x = F.relu(self.fc(x))
        x = self.fc3(x)
        return x

    def loss(self):
        return self.LDR1.loss()

class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.W = StructuredLinear(params)

        self.params = params

        if self.params.num_layers==1:
            self.b1 = Parameter(torch.Tensor(params.layer_size))
            self.W2 = Parameter(torch.Tensor(params.layer_size, params.out_size))
            self.b2 = Parameter(torch.Tensor(params.out_size))
            # Note in TF it used to be truncated normal
            torch.nn.init.normal_(self.b1,std=params.init_stddev)
            torch.nn.init.normal_(self.b2,std=params.init_stddev)
            torch.nn.init.normal_(self.W2,std=params.init_stddev)

    def forward(self, x):
        xW = self.W(x)

        if self.params.num_layers==0:
            return xW
        elif self.params.num_layers==1:
            h = F.relu(xW + self.b1)
            y = torch.matmul(h, self.W2) + self.b2
            return y
        else:
            print(('Not supported: ', params.num_layers))
            assert 0

    def loss(self):
        return 0

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.model = make_model(10, 10, 2)#(params.src_vocab, params.tgt_vocab,
            #params.N, params.d_model, params.d_ff, params.h, params.dropout)

    def forward(self, x):
        return self.model.forward(x)

    def loss(self):
        return 0

class RNN(nn.Module):
    def __init__(self, params):
        super(RNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, params):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss(self):
        return 0
