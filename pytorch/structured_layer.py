from torch.nn.parameter import Parameter
from torch_krylov import *
from torch_reconstruction import *
import torch
from torch.autograd import Variable
import torch.nn as nn

class StructuredLinear(nn.Module):
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
                fn_A, fn_B_T = self.set_mult_fns(params)
                self.fn_A = fn_A
                self.fn_B_T = fn_B_T
        else:
            print(('Not supported: ', self.params.class_type))
            assert 0

    # Assumes Stein displacement.
    def set_mult_fns(self, params):
        assert params.disp_type == 'stein'
        if params.class_type in ['toeplitz_like', 'toep_corner', 'toep_nocorn']:
            fn_A = functools.partial(Z_mult_fn, 1)
            fn_B_T = functools.partial(Z_mult_fn, -1)
        elif params.class_type == 'hankel_like':
            fn_A = functools.partial(Z_transpose_mult_fn, 1)
            fn_B_T = functools.partial(Z_mult_fn, 0)
        elif params.class_type == 'vandermonde_like':
            v = Parameter(torch.Tensor(params.layer_size))
            torch.nn.init.normal(v,std=params.init_stddev)
            self.v = v
            fn_A = functools.partial(diag_mult_fn, self.v)
            fn_B_T = functools.partial(Z_transpose_mult_fn, 0)
        elif params.class_type == 'circulant_sparsity':
            self.subdiag_f_A = Parameter(torch.Tensor(params.layer_size))
            self.subdiag_f_B = Parameter(torch.Tensor(params.layer_size))
            torch.nn.init.normal(self.subdiag_f_A,std=params.init_stddev)
            torch.nn.init.normal(self.subdiag_f_B,std=params.init_stddev)

            fn_A = functools.partial(circ_mult_fn, self.subdiag_f_A)
            fn_B_T = functools.partial(circ_mult_fn, self.subdiag_f_B)

        elif params.class_type == 'tridiagonal_corner':
            self.subdiag_f_A = Parameter(torch.Tensor(params.layer_size))
            self.subdiag_f_B = Parameter(torch.Tensor(params.layer_size))
            self.diag_A = Parameter(torch.Tensor(params.layer_size))
            self.diag_B = Parameter(torch.Tensor(params.layer_size))
            self.supdiag_A = Parameter(torch.Tensor(params.layer_size-1))
            self.supdiag_B = Parameter(torch.Tensor(params.layer_size-1))

            torch.nn.init.normal(self.subdiag_f_A,std=params.init_stddev)
            torch.nn.init.normal(self.subdiag_f_B,std=params.init_stddev)
            torch.nn.init.normal(self.diag_A,std=params.init_stddev)
            torch.nn.init.normal(self.diag_B,std=params.init_stddev)
            torch.nn.init.normal(self.supdiag_A,std=params.init_stddev)
            torch.nn.init.normal(self.supdiag_B,std=params.init_stddev)

            fn_A = functools.partial(tridiag_mult_fn, self.subdiag_f_A, self.diag_A, self.supdiag_A)
            fn_B_T = functools.partial(tridiag_mult_fn, self.subdiag_f_B, self.diag_B, self.supdiag_B)

        else:
            print(('Not supported: ', params.class_type))
            assert 0
        return fn_A, fn_B_T


    def forward(self,x):
        if self.params.class_type == 'unconstrained':
            return torch.matmul(x, self.W)
        elif self.params.class_type == 'low_rank':
            xH = torch.matmul(x, self.H)
            return torch.matmul(xH, self.G.t())
        elif self.params.class_type in ['toeplitz_like', 'vandermonde_like', 'hankel_like',
            'circulant_sparsity', 'tridiagonal_corner']:

            W = recon(self)

            return torch.matmul(x, W)
