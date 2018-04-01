import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_krylov import *
from torch_reconstruction import *

def construct_net(params):
    if params.model == 'LeNet':
        return LeNet(params)
    elif params.model == 'MLP':
        return MLP(params)
    elif params.model == 'RNN':
        return RNN(params)
    else:
        print 'Model not supported: ', params.model
        assert 0

# Assumes Stein displacement. 
def set_mult_fns(net, params):
    assert params.disp_type == 'stein'
    if params.class_type == 'toeplitz_like':
        fn_A = functools.partial(Z_transpose_mult_fn, 1)
        fn_B_T = functools.partial(Z_transpose_mult_fn, -1)
    elif params.class_type == 'hankel_like':
        fn_A = functools.partial(Z_transpose_mult_fn, 1)
        fn_B_T = functools.partial(Z_mult_fn, 0)
    elif params.class_type == 'vandermonde_like':
        v = Parameter(torch.Tensor(params.layer_size))
        torch.nn.init.normal(v,std=params.init_stddev)
        net.v = v
        fn_A = functools.partial(diag_mult_fn, net.v)
        fn_B_T = functools.partial(Z_transpose_mult_fn, 0)
    elif params.class_type == 'circulant_sparsity':
        net.subdiag_f_A = Parameter(torch.Tensor(params.layer_size))
        net.subdiag_f_B = Parameter(torch.Tensor(params.layer_size))
        torch.nn.init.normal(net.subdiag_f_A,std=params.init_stddev)
        torch.nn.init.normal(net.subdiag_f_B,std=params.init_stddev)

        fn_A = functools.partial(circ_transpose_mult_fn, net.subdiag_f_A)
        fn_B_T = functools.partial(circ_transpose_mult_fn, net.subdiag_f_B)

    elif params.class_type == 'tridiagonal_corner':
        net.subdiag_f_A = Parameter(torch.Tensor(params.layer_size))
        net.subdiag_f_B = Parameter(torch.Tensor(params.layer_size))
        net.diag_A = Parameter(torch.Tensor(params.layer_size))
        net.diag_B = Parameter(torch.Tensor(params.layer_size))
        net.supdiag_A = Parameter(torch.Tensor(params.layer_size-1))
        net.supdiag_B = Parameter(torch.Tensor(params.layer_size-1))

        torch.nn.init.normal(net.subdiag_f_A,std=params.init_stddev)
        torch.nn.init.normal(net.subdiag_f_B,std=params.init_stddev)
        torch.nn.init.normal(net.diag_A,std=params.init_stddev)
        torch.nn.init.normal(net.diag_B,std=params.init_stddev)
        torch.nn.init.normal(net.supdiag_A,std=params.init_stddev)
        torch.nn.init.normal(net.supdiag_B,std=params.init_stddev)

        fn_A = functools.partial(tridiag_transpose_mult_fn, net.subdiag_f_A, net.diag_A, net.supdiag_A)
        fn_B_T = functools.partial(tridiag_transpose_mult_fn, net.subdiag_f_B, net.diag_B, net.supdiag_B)

    else:
        print 'Not supported: ', params.class_type  
        assert 0  
    return fn_A, fn_B_T    

def structured_layer(net, x):
    if net.params.class_type == 'unconstrained':
        return torch.matmul(x, net.W)
    elif net.params.class_type == 'low_rank':
        xH = torch.matmul(x, net.H)
        return torch.matmul(xH, net.G.t())        
    elif net.params.class_type in ['toeplitz_like', 'vandermonde_like', 'hankel_like', 
        'circulant_sparsity', 'tridiagonal_corner']:
        W = krylov_recon(net.params, net.G, net.H, net.fn_A, net.fn_B_T)
        
        # NORMALIZE W

        return torch.matmul(x, W)

    else:
        print 'Not supported: ', params.class_type  
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
        print 'Not supported: ', params.class_type  
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
            #thing = torch.matmul(x, self.H)
            #y = torch.matmul(thing, self.G.t())
            return xW
        elif self.params.num_layers==1:
            h = F.relu(xW + self.b1)
            y = torch.matmul(h, self.W2) + self.b2
            return y
        else:
            print 'Not supported: ', params.num_layers
            assert 0

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