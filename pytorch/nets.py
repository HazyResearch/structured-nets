import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import copy
from inspect import signature
import sys
import numpy as np
sys.path.insert(0, '../../krylov/')
# sys.path.insert(0, '../../pytorch/attention/')

# from torch_krylov import *
# from torch_reconstruction import *
# from attention import *
# import toeplitz_gpu as toep
# import krylov_multiply as subd
import LDR as ldr
#from krylov_multiply import *
# from structured_layer import StructuredLinear # this is already in attention
import structured_layer as structured
# TODO fix the 'import *'s


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def construct_net(params):
    if params.model == 'LeNet':
        return LeNet(params)
    elif params.model == 'Softmax':
        return Softmax(params)
    elif params.model == 'SHL':
        return SHL(params)
    elif params.model == 'LDRNet':
        return LDRNet(params)
    elif params.model == 'LDRfat':
        return LDRFat(params)
    elif params.model == 'LDRLDR':
        return LDRLDR(params)
    elif params.model == 'LDRLDR2':
        return LDRLDR2(params)
    elif params.model == 'RNN':
        return RNN(params)
    elif params.model == 'Attention':
        return Attention(params)
    elif params.model == 'CNN':
        return CNN(params)
    elif params.model == 'CNNPool':
        return CNNPool(params)
    else:
        print(('Model not supported: ', params.model))
        assert 0

def construct_model(cls, in_size, out_size, args):
    args_fn = cls.args
    options = {param: vars(args)[param]
                for param in signature(args_fn).parameters}
    return cls(in_size=in_size, out_size=out_size, **options)

class ArghModel(nn.Module):
    # def __init__(self, in_size, out_size, **options):
    def __init__(self, **options):
        """"
        options: dictionary of options/params that args() accepts
        If the model if constructed with construct_model(), the options will contain defaults based on its args() function
        """
        super().__init__()

        # self.in_size = in_size
        # self.out_size = out_size
        self.__dict__.update(**options)
        self.init()

    def init(self):
        raise NotImplementedError()

    def args(**kwargs):
        """
        Empty function whose signature contains parameters and defaults for the class
        """
        raise NotImplementedError()
        # self.__dict__.update(**kwargs)

    def name():
        """
        Short string summarizing the main parameters of the class
        Used to construct a unique identifier for an experiment
        """
        raise NotImplementedError()

    def loss(self):
        """
        Loss function for the model's parameters (e.g. regularization)
        """
        return 0

    method_map = None # TODO move method map here to help with id()



# Pytorch tutorial lenet variant
class PTLeNet(nn.Module):
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

# single channel LeNet
class LeNet(nn.Module):
    def __init__(self, params):
        super(LeNet, self).__init__()
        self.d = int(np.sqrt(params.layer_size))
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        # self.fc = nn.Linear(self.d**2, self.d**2)
        self.W = StructuredLinear(params)
        self.b = Parameter(torch.zeros(self.d**2))
        self.logits = nn.Linear(self.d**2, 10)

    def forward(self, x):
        x = x.view(-1, 1, self.d, self.d)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.d**2)
        x = F.relu(self.W(x) + self.b)
        x = self.logits(x)
        return x

    def loss(self):
        return 0

# Simple 3 layer CNN with pooling
class CNNPool(nn.Module):
    def __init__(self, params):
        super(CNNPool, self).__init__()
        self.channels = 3
        fc_size = 512
        self.conv1 = nn.Conv2d(3, self.channels, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(self.channels/4*1024, fc_size)
        self.logits = nn.Linear(fc_size,10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, self.channels/4*1024)

        x = F.relu(self.fc(x))
        x = self.logits(x)
        return x

    def loss(self):
        return 0


# Simple 2 layer CNN: convolution channels, FC, softmax
class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()
        self.noconv = False
        self.pool = True
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        if self.noconv:
            self.conv1 = nn.Linear(3*1024, 3*1024)
        else:
            self.conv1 = nn.Conv2d(3, 3, 5, padding=2)

        if self.pool:
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(768, 512)
            self.logits = nn.Linear(512,10)
        else:
            self.fc = nn.Linear(3*1024, 512)
            self.logits = nn.Linear(512, 10)

    def forward(self, x):
        if self.noconv:
            x = F.relu(self.conv1(x))
        else:
            x = x.view(-1, 3, 32, 32)
            x = F.relu(self.conv1(x))
            x = x.view(-1, 3*1024)

        if self.pool:
            x = x.view(-1, 3, 32, 32)
            x = self.pool(x)
            x = x.view(-1, 768)

        x = F.relu(self.fc(x))
        x = self.logits(x)
        return x

    def loss(self):
        return 0

# LDR layer (single weight matrix), followed by FC and softmax
class LDRFat(nn.Module):
    def __init__(self, params):
        super(LDRFat, self).__init__()

        fc_size = 512

        self.params = params

        self.W = StructuredLinear(class_type=params.class_type, layer_size=3072, init_stddev=params.init_stddev, r=params.r)
        self.fc = nn.Linear(3072, fc_size)
        self.logits = nn.Linear(fc_size, 10)

    def forward(self, x):
        x = self.W(x)
        x = F.relu(self.fc(x))
        x = self.logits(x)
        return x

    def loss(self):
        lamb = 0.0001
        if self.params.class_type == 'unconstrained':
            return 0 #TODO should probably compare against L2 regularized
        return lamb*torch.sum(torch.abs(self.W.G)) + lamb*torch.sum(torch.abs(self.W.H))



# LDR layer with channels, followed by FC and softmax
class LDRNet(nn.Module):
    def __init__(self, params):
        super(LDRNet, self).__init__()
        self.n = 1024
        channels = 3
        fc_size = 512

        self.LDR1 = ldr.LDR(params.class_type, 3, 3, params.r, self.n, bias=True)
        # self.LDR2 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.LDR3 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.LDR4 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.b1 = Parameter(torch.Tensor(params.layer_size))
        # torch.nn.init.normal_(self.b1,std=params.init_stddev)
        # self.W1 = Parameter(torch.Tensor(3*1024, fc_size))
        self.fc = nn.Linear(3*self.n, fc_size)
        self.logits = nn.Linear(fc_size, 10)

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
        x = self.logits(x)
        return x

    def loss(self):
        return self.LDR1.loss()

# LDR layer with 3 channels, followed by another LDR layer, then softmax
class LDRLDR(nn.Module):
    def __init__(self, params):
        super(LDRLDR, self).__init__()
        self.channels = False

        self.n = 1024
        fc_size = 512

        rank1 = 48
        rank2 = 16
        class1 = 'toep_nocorn'
        class2 = 'toep_nocorn'

        if self.channels:
            self.LDR1 = ldr.LDR(class1, 3, 3, rank1, self.n, bias=True)
        else:
            self.LDR1 = StructuredLinear(class_type=class1, layer_size=3*1024, init_stddev=params.init_stddev, r=rank1)

        self.LDR211 = StructuredLinear(class_type=class2, layer_size=fc_size, init_stddev=params.init_stddev, r=rank2)
        self.LDR212 = StructuredLinear(class_type=class2, layer_size=fc_size, init_stddev=params.init_stddev, r=rank2)
        self.LDR221 = StructuredLinear(class_type=class2, layer_size=fc_size, init_stddev=params.init_stddev, r=rank2)
        self.LDR222 = StructuredLinear(class_type=class2, layer_size=fc_size, init_stddev=params.init_stddev, r=rank2)
        self.LDR231 = StructuredLinear(class_type=class2, layer_size=fc_size, init_stddev=params.init_stddev, r=rank2)
        self.LDR232 = StructuredLinear(class_type=class2, layer_size=fc_size, init_stddev=params.init_stddev, r=rank2)
        self.b = Parameter(torch.zeros(fc_size))
        self.logits = nn.Linear(fc_size, 10)

    def forward(self, x):
        if self.channels:
            x = x.view(-1, 3, 1024)
            x = x.transpose(0,1).contiguous().view(3, -1, self.n)
            x = F.relu(self.LDR1(x))
        else:
            x = F.relu(self.LDR1(x))
            x = x.view(-1, 3, 1024)
            x = x.transpose(0,1).contiguous().view(3, -1, self.n)
        x11 = x[0][:,:512]
        x12 = x[0][:,512:]
        x21 = x[1][:,:512]
        x22 = x[1][:,512:]
        x31 = x[2][:,:512]
        x32 = x[2][:,512:]
        # x = x.transpose(0,1) # swap batches and channels axis
        # x = x.contiguous().view(-1, 3*self.n)
        # x = F.relu(self.LDR21(x1) + self.LDR22(x2) + self.LDR23(x3) + self.b)
        x = F.relu(self.LDR211(x11) + self.LDR212(x12) + self.LDR221(x21) + self.LDR222(x22) + self.LDR231(x31) + self.LDR232(x32) + self.b)
        x = self.logits(x)
        return x

    def loss(self):
        if self.channels:
            return self.LDR1.loss()
        else:
            lamb = 0.0001
            # return 0
            return lamb*torch.sum(torch.abs(self.LDR1.G)) + lamb*torch.sum(torch.abs(self.LDR1.H))


class LDRLDR2(nn.Module):
    def __init__(self, params):
        super(LDRLDR2, self).__init__()

        self.n = 1024
        self.channels = 4
        self.fc_size = 512

        rank1 = 48
        rank2 = 16
        class1 = 'subdiagonal'
        class2 = 'subdiagonal'

        self.LDR1 = StructuredLinear(class_type=class1, layer_size=self.channels*self.n, init_stddev=params.init_stddev, r=rank1, bias=True)
        self.LDR2 = StructuredLinear(class_type=class2, layer_size=self.channels*self.n, init_stddev=params.init_stddev, r=rank2, bias=True)
        self.logits = nn.Linear(self.fc_size, 10)

    def forward(self, x):
        batch_size, n = x.shape[0], x.shape[1]
        x = torch.cat((x, torch.zeros(batch_size, self.channels*self.n-n).cuda()), dim=-1)
        x = F.relu(self.LDR1(x))
        x = F.relu(self.LDR2(x))
        x = x[:,:self.fc_size]
        x = self.logits(x)
        return x

    def loss(self):
        lamb = 0.0001
        # return 0
        return lamb*torch.sum(torch.abs(self.LDR1.G)) + lamb*torch.sum(torch.abs(self.LDR1.H))


class Softmax(nn.Module):
    def __init__(self, params):
        super(SHL, self).__init__()
        self.W = StructuredLinear(params)
        self.params = params

    def forward(self, x):
        xW = self.W(x)
        return xW

    def name():
        return ''

    def loss(self):
        return 0

class SHL(ArghModel):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def init(self):
        if self.layer_size == -1:
            self.layer_size = self.in_size

        # W = structured.StructuredLinear(class_type=self.class_type, layer_size=self.layer_size, r=self.r, bias=self.bias)
        W = structured.class_map[self.class_type](layer_size=self.layer_size, r=self.r, bias=self.bias)

        b1 = Parameter(torch.Tensor(self.layer_size))
        W2 = Parameter(torch.Tensor(self.layer_size, self.out_size))
        b2 = Parameter(torch.Tensor(self.out_size))
        # Note in TF it used to be truncated normal
        torch.nn.init.normal_(b1,std=self.init_stddev)
        torch.nn.init.normal_(b2,std=self.init_stddev)
        torch.nn.init.normal_(W2,std=self.init_stddev)

        self.W = W
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        # return {'W': W, 'b1': b1, 'W2': W2, 'b2': b2}

    def args(class_type='unconstrained', layer_size=-1, r=1, init_stddev=0.01, bias=True):
        pass

    def name(self):
        return self.W.name()

    def forward(self, x):
        xW = self.W(x)

        h = F.relu(xW + self.b1)
        y = torch.matmul(h, self.W2) + self.b2
        return y

