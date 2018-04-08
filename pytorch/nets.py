import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_krylov import *
from torch_reconstruction import *
import sys
sys.path.insert(0, '../../krylov/')
sys.path.insert(0, '../../pytorch/attention/')
from attention import *
#from krylov_multiply import *
import copy

def construct_net(params):
    if params.model == 'LeNet':
        return LeNet(params)
    elif params.model == 'MLP':
        return MLP(params)
    elif params.model == 'RNN':
        return RNN(params)
    elif params.model == 'Attention':
        return Attention(params)
    else:
        print('Model not supported: ', params.model)
        assert 0
  
def structured_layer(net, x):
	if net.params.class_type == 'unconstrained':
		return torch.matmul(x, net.W)
	elif net.params.class_type == 'low_rank':
		xH = torch.matmul(x, net.H)
		return torch.matmul(xH, net.G.t())        
	elif net.params.class_type in ['toeplitz_like', 'vandermonde_like', 'hankel_like', 
        'circulant_sparsity', 'tridiagonal_corner']:
		#print('krylov fast')
		#print('net.H: ', net.H.t())
		#print('x: ', x)
		#print(KB)
		#return krylov_multiply_fast(net.subdiag_f_A[1:], net.G.t(), krylov_transpose_multiply_fast(net.subdiag_f_B[1:], net.H.t(), x))
		W = krylov_recon(net.params, net.G, net.H, net.fn_A, net.fn_B_T)
		# NORMALIZE W
		return torch.matmul(x, W)
	else:
		print('Not supported: ', params.class_type)  
		assert 0 

def set_structured_W(net, params):
    if params.class_type == 'unconstrained':
        net.W = Parameter(torch.Tensor(params.layer_size, params.layer_size))
        torch.nn.init.normal(net.W, std=params.init_stddev)
    elif params.class_type in ['low_rank', 'toeplitz_like', 'vandermonde_like', 'hankel_like', 
        'circulant_sparsity', 'tridiagonal_corner']:
        net.G = Parameter(torch.Tensor(params.layer_size, params.r))
        net.H = Parameter(torch.Tensor(params.layer_size, params.r))
        torch.nn.init.normal(net.G, std=params.init_stddev)
        torch.nn.init.normal(net.H, std=params.init_stddev)

        if params.class_type != 'low_rank':
            fn_A, fn_B_T = set_mult_fns(net, params)
            net.fn_A = fn_A
            net.fn_B_T = fn_B_T
    else:
        print('Not supported: ', params.class_type)  
        assert 0

class LeNet(nn.Module):
    def __init__(self, params):
        super(LeNet, self).__init__()
        self.W1 = get_structured_W(params)
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
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
            print('Not supported: ', params.num_layers)
            assert 0

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.model = make_model(10, 10, 2)#(params.src_vocab, params.tgt_vocab, 
            #params.N, params.d_model, params.d_ff, params.h, params.dropout)

    def forward(self, x):
        return self.model.forward(x)

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
