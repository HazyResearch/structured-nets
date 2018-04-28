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
    elif params.model == 'RNN':
        return RNN(params)
    elif params.model == 'Attention':
        return Attention(params)
    elif params.model == 'Test':
        return TestCifar(params)
    else:
        print(('Model not supported: ', params.model))
        assert 0

def structured_layer(net, x):
    if net.params.class_type == 'unconstrained':
        return torch.matmul(x, net.W)
    elif net.params.class_type == 'low_rank':
        xH = torch.matmul(x, net.H.t())
        return torch.matmul(xH, net.G)
    elif net.params.class_type in ['toep_corner', 'toep_nocorn']:
        # test that it matches slow version
        # W = krylov_recon(net.params.r, net.params.layer_size, net.G, net.H, net.fn_A, net.fn_B_T)
        # ans = toep.toeplitz_mult(net.G, net.H, x, net.cycle)
        # print(torch.norm(ans-torch.matmul(x, W.t())))
        # K = toep.krylov_construct(1, net.G[0].squeeze(), net.params.layer_size)
        # KT = toep.krylov_construct(-1, net.H[0].squeeze(), net.params.layer_size)
        # W1 = torch.matmul(K, KT.t())
        # ans2 = toep.toeplitz_mult_slow(net.G, net.H, x, net.cycle)
        # print(torch.norm(ans-torch.matmul(x, W1.t())))
        return toep.toeplitz_mult(net.G, net.H, x, net.cycle)
        # return toep.toeplitz_mult(net.G.t(), net.H.t(), x, net.cycle)
    elif net.params.class_type == 'subdiagonal':
        return subd.subd_mult(net.subd_A, net.subd_B, net.G, net.H, x)
        # return subd.krylov_multiply_fast(net.subd_A, net.G, subd.krylov_transpose_multiply_fast(net.subd_B, net.H, x))
    elif net.params.class_type in ['toeplitz_like', 'vandermonde_like', 'hankel_like',
        'circulant_sparsity', 'tridiagonal_corner']:
        #print('krylov fast')
        #print('net.H: ', net.H)
        #print('x: ', x)
        #print(KB)
        #return krylov_multiply_fast(net.subdiag_f_A[1:], net.G, krylov_transpose_multiply_fast(net.subdiag_f_B[1:], net.H, x))
        W = krylov_recon(net.params.r, net.params.layer_size, net.G, net.H, net.fn_A, net.fn_B_T)
        # NORMALIZE W
        return torch.matmul(x, W.t())
    else:
        print(('Not supported: ', net.params.class_type))
        assert 0

def set_structured_W(net, params):
    if params.class_type == 'unconstrained':
        net.W = Parameter(torch.Tensor(params.layer_size, params.layer_size))
        torch.nn.init.normal(net.W, std=params.init_stddev)
    elif params.class_type in ['low_rank', 'toeplitz_like', 'toep_corner', 'toep_nocorn', 'subdiagonal', 'vandermonde_like', 'hankel_like',
        'circulant_sparsity', 'tridiagonal_corner']:
        net.G = Parameter(torch.Tensor(params.r, params.layer_size))
        net.H = Parameter(torch.Tensor(params.r, params.layer_size))
        torch.nn.init.normal(net.G, std=params.init_stddev)
        torch.nn.init.normal(net.H, std=params.init_stddev)

        if params.class_type == 'low_rank':
            pass
        elif params.class_type == 'toep_corner':
            net.cycle = True
            fn_A, fn_B_T = StructuredLinear.set_mult_fns(net, params)
            net.fn_A = fn_A
            net.fn_B_T = fn_B_T
        elif params.class_type == 'toep_nocorn':
            net.cycle = False
        elif params.class_type == 'subdiagonal':
            net.subd_A = Parameter(torch.ones(params.layer_size))
            net.subd_B = Parameter(torch.ones(params.layer_size))
        else:
            fn_A, fn_B_T = StructuredLinear.set_mult_fns(net, params)
            net.fn_A = fn_A
            net.fn_B_T = fn_B_T

    else:
        print(('Not supported: ', params.class_type))
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

class TestCifar(nn.Module):
    def __init__(self, params):
        super(TestCifar, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = nn.Conv2d(3, 3, 5, padding=2)
        self.fc = nn.Linear(3*1024, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 3*1024)
        x = F.relu(self.fc(x))
        x = self.fc3(x)
        return x

    def loss(self):
        return 0

class LDRNet(nn.Module):
    def __init__(self, params):
        super(LDRNet, self).__init__()
        self.n = params.layer_size

        channels = 1

        self.LDR1 = ldr.LDR(params.class_type, 3, 3, params.r, params.layer_size, bias=True)
        # self.LDR2 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.LDR3 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.LDR4 = ldr.LDR(params.class_type, 1, 1, params.r, params.layer_size)
        # self.b1 = Parameter(torch.Tensor(params.layer_size))
        # torch.nn.init.normal(self.b1,std=params.init_stddev)
        # self.W1 = Parameter(torch.Tensor(3*1024, 84))
        self.fc = nn.Linear(3*self.n, 84)
        self.fc3 = nn.Linear(84, 10)

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
        set_structured_W(self, params)

        self.params = params

        if self.params.num_layers==1:
            self.b1 = Parameter(torch.Tensor(params.layer_size))
            self.W2 = Parameter(torch.Tensor(params.layer_size, params.out_size))
            self.b2 = Parameter(torch.Tensor(params.out_size))
            # Note in TF it used to be truncated normal
            torch.nn.init.normal(self.b1,std=params.init_stddev)
            torch.nn.init.normal(self.b2,std=params.init_stddev)
            torch.nn.init.normal(self.W2,std=params.init_stddev)

    def forward(self, x):
        xW = structured_layer(self, x)

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
