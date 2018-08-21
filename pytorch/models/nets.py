from inspect import signature
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import structure.LDR as ldr
import structure.layer as sl
from utils import descendants


# create a map from structured class names to the Python class
class_map = {}
for cls in descendants(sl.StructuredLinear):
    if cls.class_type is None: continue
    class_map[cls.class_type] = cls
    class_map[cls.abbrev] = cls


def construct_model(cls, in_size, out_size, args):
    args_fn = cls.args
    options = {param: vars(args)[param]
                for param in signature(args_fn).parameters}
    return cls(in_size=in_size, out_size=out_size, **options)

class ArghModel(nn.Module):
    def __init__(self, in_size, out_size, **options):
        """"
        options: dictionary of options/params that args() accepts
        If the model if constructed with construct_model(), the options will contain defaults based on its args() function
        """
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.__dict__.update(**options)
        self.reset_parameters()

    def args():
        """
        Empty function whose signature contains parameters and defaults for the class
        """
        pass

    def reset_parameters(self):
        pass

    def name(self):
        """
        Short string summarizing the main parameters of the class
        Used to construct a unique identifier for an experiment
        """
        return ''

    def loss(self):
        """
        Model-specific loss function (e.g. per-parameter regularization)
        """
        return 0



# Pytorch tutorial lenet variant
class Lenet(ArghModel):
    def reset_parameters(self):
        # super().__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(ArghModel):
    """
    Single channel net where the dense last layer has same dimensions as input
    """
    def args(class_type='unconstrained', layer_size=-1, r=1, bias=True): pass
    def reset_parameters(self):
        if self.layer_size == -1:
            self.layer_size = self.in_size
        assert self.layer_size == self.in_size
        self.d = int(np.sqrt(self.layer_size))
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.W = class_map[self.class_type](layer_size=self.layer_size, r=self.r, bias=self.bias)
        self.logits = nn.Linear(self.layer_size, self.out_size)

    def name(self):
        return self.W.name()

    def forward(self, x):
        x = x.view(-1, 1, self.d, self.d)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.layer_size)
        x = F.relu(self.W(x))
        x = self.logits(x)
        return x

class CNNColor(ArghModel):
    """
    3 channel net where the dense last layer has same dimensions as input
    """
    def args(class_type='unconstrained', layer_size=-1, r=1, bias=True): pass
    def reset_parameters(self):
        self.layer_size = int(self.in_size/3)
        self.d = int(np.sqrt(self.layer_size))
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.W = class_map[self.class_type](layer_size=self.layer_size, r=self.r, bias=self.bias)
        self.logits = nn.Linear(self.layer_size, self.out_size)

    def name(self):
        return self.W.name()

    def forward(self, x):
        x = x.view(-1, 3, self.d, self.d)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.layer_size)
        x = F.relu(self.W(x))
        x = self.logits(x)
        return x

class CNNPool(ArghModel):
    """
    Simple 3 layer CNN with pooling for cifar10
    """
    def name(self):
        return str(self.channels) + 'pool'

    def args(channels=3, fc_size=512): pass
    def reset_parameters(self):
        self.conv1 = nn.Conv2d(3, self.channels, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(self.channels*1024//4, self.fc_size)
        self.logits = nn.Linear(self.fc_size,10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, self.channels*1024//4)

        x = F.relu(self.fc(x))
        x = self.logits(x)
        return x


class TwoLayer(ArghModel):
    """
    Simple 2 hidden layer network: convolution channels or FC, FC, softmax
    """
    def name(self):
        return "3conv"

    def args(conv=True): pass
    def reset_parameters(self):
        if self.conv:
            self.conv1 = nn.Conv2d(3, 3, 5, padding=2)
        else:
            self.conv1 = nn.Linear(3*1024, 3*1024)

        self.fc = nn.Linear(3*1024, 512)
        self.logits = nn.Linear(512, 10)

    def forward(self, x):
        if self.conv:
            x = x.view(-1, 3, 32, 32)
            x = F.relu(self.conv1(x))
            x = x.view(-1, 3*1024)
        else:
            x = F.relu(self.conv1(x))

        x = F.relu(self.fc(x))
        x = self.logits(x)
        return x

    def loss(self):
        return 0

class WLDRFC(ArghModel):
    """
    LDR layer (single weight matrix), followed by FC and softmax
    """
    def name(self):
        return self.W.name()+'u'

    def args(class_type='unconstrained', layer_size=-1, r=1, fc_size = 512): pass
    def reset_parameters(self):
        if self.layer_size == -1: self.layer_size = self.in_size
        self.W = class_map[self.class_type](layer_size=self.layer_size, r=self.r)
        self.fc = nn.Linear(3*1024, self.fc_size)
        self.logits = nn.Linear(self.fc_size, 10)

    def forward(self, x):
        x = self.W(x)
        x = F.relu(self.fc(x))
        x = self.logits(x)
        return x

    def loss(self):
        return self.W.loss()


class LDRFC(ArghModel):
    """
    LDR layer with channels, followed by FC and softmax
    """
    def name(self):
        return self.W.name()+'u'

    def args(class_type='t', r=1, channels=3, fc_size=512): pass
    def reset_parameters(self):
        self.n = 1024

        self.LDR1 = ldr.LDR(self.class_type, 3, self.channels, self.r, self.n, bias=True)
        self.fc = nn.Linear(self.channels*self.n, self.fc_size)
        self.logits = nn.Linear(self.fc_size, 10)

    def forward(self, x):
        x = x.view(-1, 3, 1024)
        x = x.transpose(0,1).contiguous().view(3, -1, self.n)
        x = F.relu(self.LDR1(x))
        x = x.transpose(0,1) # swap batches and channels axis
        x = x.contiguous().view(-1, self.channels*self.n)
        x = F.relu(self.fc(x))
        x = self.logits(x)
        return x

    def loss(self):
        return self.LDR1.loss()

class LDRLDR(ArghModel):
    """
    LDR layer (either 3 channels or one wide matrix), followed by another LDR layer, then softmax
    intended for 3-channel images of size 1024 (e.g. CIFAR-10)
    """
    def name(self):
        # w = 'wide' if not self.channels else ''
        return self.LDR1.name() + self.LDR211.name()

    def args(class1='toeplitz', class2='toeplitz', channels=False, rank1=48, rank2=16): pass
    def reset_parameters(self):
        self.n = 1024
        self.fc_size = self.n // 2

        if self.channels:
            self.LDR1 = ldr.LDR(self.class1, 3, 3, self.rank1, self.n)
        else:
            self.LDR1 = class_map[self.class1](layer_size=3*self.n, r=self.rank1)

        self.LDR211 = class_map[self.class2](layer_size=self.fc_size, r=self.rank2)
        self.LDR212 = class_map[self.class2](layer_size=self.fc_size, r=self.rank2)
        self.LDR221 = class_map[self.class2](layer_size=self.fc_size, r=self.rank2)
        self.LDR222 = class_map[self.class2](layer_size=self.fc_size, r=self.rank2)
        self.LDR231 = class_map[self.class2](layer_size=self.fc_size, r=self.rank2)
        self.LDR232 = class_map[self.class2](layer_size=self.fc_size, r=self.rank2)
        self.b = Parameter(torch.zeros(self.fc_size))
        self.logits = nn.Linear(self.fc_size, 10)

    def forward(self, x):
        if self.channels:
            x = x.view(-1, 3, self.n)
            x = x.transpose(0,1).contiguous().view(3, -1, self.n)
            x = F.relu(self.LDR1(x))
        else:
            x = F.relu(self.LDR1(x))
            x = x.view(-1, 3, self.n)
            x = x.transpose(0,1).contiguous().view(3, -1, self.n)
        x11 = x[0][:,:self.fc_size]
        x12 = x[0][:,self.fc_size:]
        x21 = x[1][:,:self.fc_size]
        x22 = x[1][:,self.fc_size:]
        x31 = x[2][:,:self.fc_size]
        x32 = x[2][:,self.fc_size:]
        x = F.relu(self.LDR211(x11) + self.LDR212(x12) + self.LDR221(x21) + self.LDR222(x22) + self.LDR231(x31) + self.LDR232(x32) + self.b)
        x = self.logits(x)
        return x

    def loss(self):
        return self.LDR1.loss()


class LDRLDR2(ArghModel):
    """
    Same as LDRLDR but use larger matrix to represent rectangular LDR
    """
    def name(self):
        return self.LDR1.name() + self.LDR2.name()

    def args(class1='toeplitz', class2='toeplitz', layer_size=-1, channels=3, fc_size=512, rank1=48, rank2=16): pass
    def reset_parameters(self):
        if self.layer_size == -1:
            self.layer_size = self.in_size
        self.n = self.layer_size

        self.LDR1 = class_map[self.class1](layer_size=self.channels*self.n, r=self.rank1, bias=True)
        self.LDR2 = class_map[self.class2](layer_size=self.channels*self.n, r=self.rank2, bias=True)
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
        return self.LDR1.loss()


class SL(ArghModel):
    """
    Single layer linear model (for synthetic regression tests)
    """
    def name(self):
        return self.W.name()

    def args(class_type='unconstrained', layer_size=-1, r=1, bias=False): pass
    def reset_parameters(self):
        if self.layer_size == -1:
            self.layer_size = self.in_size
        self.W = class_map[self.class_type](layer_size=self.layer_size, r=self.r, bias=self.bias)

    def forward(self, x):
        return self.W(x)

class SHL(SL):
    """
    Single hidden layer
    """
    def args(class_type='unconstrained', layer_size=-1, r=1, bias=True): pass
    def reset_parameters(self):
        super().reset_parameters()
        self.W2 = nn.Linear(self.layer_size, self.out_size)

    def forward(self, x):
        return self.W2(F.relu(self.W(x)))

class MLP(ArghModel):
    """
    Multi layer fully connected net.
    """
    def args(class_type='unconstrained', layer_size=-1, r=1, bias=False, num_layers=1): pass
    def reset_parameters(self):
        if self.layer_size == -1:
            self.layer_size = self.in_size
        layers = []
        for layer in range(self.num_layers):
            layers.append(class_map[self.class_type](layer_size=self.layer_size, r=self.r, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.W2 = nn.Linear(self.layer_size, self.out_size)

    def forward(self, x):
        output = F.relu(self.layers[0](x))
        for i in range(self.num_layers-1):
            output = F.relu(self.layers[i](output))
        return self.W2(output)
