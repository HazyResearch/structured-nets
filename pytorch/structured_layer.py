from torch.nn.parameter import Parameter
from torch_krylov import *
from torch_reconstruction import *
import torch
from torch.autograd import Variable
import torch.nn as nn

class StructuredLinear(nn.module):
    def __init__(self, params):
        super(StructuredLinear,self).__init__()
        self.params = params
        if self.params.class_type == 'unconstrained':
            self.W = Parameter(torch.Tensor(params.layer_size, params.layer_size))
            torch.nn.init.normal(self.W, std=params.init_stddev)
        elif self.params.class_type in ['low_rank', 'toeplitz_like', 'vandermonde_like', 'hankel_like', 
            'circulant_sparsity', 'tridiagonal_corner']:
            self.G = Parameter(torch.Tensor(params.layer_size, params.r))
            self.H = Parameter(torch.Tensor(params.layer_size, params.r))
            torch.nn.init.normal(self.G, std=params.init_stddev)
            torch.nn.init.normal(self.H, std=params.init_stddev)

            if self.params.class_type != 'low_rank':
                fn_A, fn_B_T = set_mult_fns(self, params)
                self.fn_A = fn_A
                self.fn_B_T = fn_B_T
        else:
            print 'Not supported: ', self.params.class_type  
            assert 0

    def forward(self,x):
        if self.params.class_type == 'unconstrained':
            return torch.matmul(x, self.W)
        elif self.params.class_type == 'low_rank':
            xH = torch.matmul(x, self.H)
            return torch.matmul(xH, self.G.t())        
        elif self.params.class_type in ['toeplitz_like', 'vandermonde_like', 'hankel_like', 
            'circulant_sparsity', 'tridiagonal_corner']:
            W = krylov_recon(self.params, self.G, self.H, self.fn_A, self.fn_B_T)
            
            # NORMALIZE W

            return torch.matmul(x, self.W)
