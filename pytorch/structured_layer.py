import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch_krylov import *
from torch_reconstruction import *
import toeplitz_gpu as toep
import krylov_multiply as subd

class StructuredLinear(nn.Module):
    def __init__(self, params):
        super(StructuredLinear,self).__init__()
        self.params = params
        if self.params.class_type == 'unconstrained':
            self.W = Parameter(torch.Tensor(params.layer_size, params.layer_size))
            torch.nn.init.normal_(self.W, std=params.init_stddev)
        elif self.params.class_type in ['low_rank', 'toeplitz_like', 'vandermonde_like', 'hankel_like',
                                        'circulant_sparsity', 'tridiagonal_corner', 'toep_corner', 'toep_nocorn', 'subdiagonal']:
            self.G = Parameter(torch.Tensor(params.r, params.layer_size))
            self.H = Parameter(torch.Tensor(params.r, params.layer_size))
            torch.nn.init.normal_(self.G, std=params.init_stddev)
            torch.nn.init.normal_(self.H, std=params.init_stddev)

            if self.params.class_type == 'low_rank':
                pass
            elif self.params.class_type == 'toep_corner':
                self.cycle = True
            elif self.params.class_type == 'toep_nocorn':
                self.cycle = False
            elif self.params.class_type == 'subdiagonal':
                self.subd_A = Parameter(torch.ones(self.params.layer_size))
                self.subd_B = Parameter(torch.ones(self.params.layer_size))
            else:
                fn_A, fn_B_T = self.set_mult_fns(self.params)
                self.fn_A = fn_A
                self.fn_B_T = fn_B_T
        else:
            print((f"{self.__class__.__name__} does not support {self.params.class_type}"))
            assert 0

    # Assumes Stein displacement.
    def set_mult_fns(self, params):
        assert params.disp_type == 'stein'
        if params.class_type in ['toeplitz_like', 'toep_corner', 'toep_nocorn']:
            fn_A = functools.partial(Z_mult_fn, 1)
            fn_B_T = functools.partial(Z_mult_fn, -1)
        # TODO: operators for hankel and vandermonde have not been checked for transpose consistency
        elif params.class_type == 'hankel_like':
            fn_A = functools.partial(Z_transpose_mult_fn, 1)
            fn_B_T = functools.partial(Z_mult_fn, 0)
        elif params.class_type == 'vandermonde_like':
            v = Parameter(torch.Tensor(params.layer_size))
            torch.nn.init.normal_(v,std=params.init_stddev)
            self.v = v
            fn_A = functools.partial(diag_mult_fn, self.v)
            fn_B_T = functools.partial(Z_transpose_mult_fn, 0)
        elif params.class_type == 'circulant_sparsity':
            self.subdiag_f_A = Parameter(torch.Tensor(params.layer_size))
            self.subdiag_f_B = Parameter(torch.Tensor(params.layer_size))
            torch.nn.init.normal_(self.subdiag_f_A,std=params.init_stddev)
            torch.nn.init.normal_(self.subdiag_f_B,std=params.init_stddev)

            fn_A = functools.partial(circ_mult_fn, self.subdiag_f_A)
            fn_B_T = functools.partial(circ_mult_fn, self.subdiag_f_B)

        elif params.class_type == 'tridiagonal_corner':
            self.subdiag_f_A = Parameter(torch.Tensor(params.layer_size))
            self.subdiag_f_B = Parameter(torch.Tensor(params.layer_size))
            self.diag_A = Parameter(torch.Tensor(params.layer_size))
            self.diag_B = Parameter(torch.Tensor(params.layer_size))
            self.supdiag_A = Parameter(torch.Tensor(params.layer_size-1))
            self.supdiag_B = Parameter(torch.Tensor(params.layer_size-1))

            torch.nn.init.normal_(self.subdiag_f_A,std=params.init_stddev)
            torch.nn.init.normal_(self.subdiag_f_B,std=params.init_stddev)
            torch.nn.init.normal_(self.diag_A,std=params.init_stddev)
            torch.nn.init.normal_(self.diag_B,std=params.init_stddev)
            torch.nn.init.normal_(self.supdiag_A,std=params.init_stddev)
            torch.nn.init.normal_(self.supdiag_B,std=params.init_stddev)

            fn_A = functools.partial(tridiag_mult_fn, self.subdiag_f_A, self.diag_A, self.supdiag_A)
            fn_B_T = functools.partial(tridiag_mult_fn, self.subdiag_f_B, self.diag_B, self.supdiag_B)

        else:
            print(('Not supported: ', params.class_type))
            assert 0
        return fn_A, fn_B_T


    def forward(self, x):
        if self.params.class_type == 'unconstrained':
            return torch.matmul(x, self.W)
        elif self.params.class_type == 'low_rank':
            xH = torch.matmul(x, self.H.t())
            return torch.matmul(xH, self.G)
        elif self.params.class_type in ['toep_corner', 'toep_nocorn']:
            return toep.toeplitz_mult(self.G, self.H, x, self.cycle)
        elif self.params.class_type == 'subdiagonal':
            return subd.subd_mult(self.subd_A, self.subd_B, self.G, self.H, x)
        elif self.params.class_type in ['toeplitz_like', 'vandermonde_like', 'hankel_like',
            'circulant_sparsity', 'tridiagonal_corner']:

            W = recon(self)

            return torch.matmul(x, W.t())
