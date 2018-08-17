import torch
import functools
import numpy as np
from torch.autograd import Variable
import time

# Down shift
def Z_mult_fn(f, x):
    return torch.cat((f * x[-1], x[:-1]))

# Up shift
def Z_transpose_mult_fn(f, x):
    #print('x[1:]: ', x[1:])
    #print('f*x[0]: ', f*x[0])
    #return torch.cat((x[1:], torch.FloatTensor([f * x[0]])))
    return torch.cat((x[1:], f*x[0]))

# Diagonal multiplication
def diag_mult_fn(diag, x):
    return diag * x

# Circulant sparsity pattern operators
def circ_mult_fn(subdiag_f, x):
    # note: f corresponds to last element instead of first
    y = torch.cat((x[-1], x[:-1]))
    return y * subdiag_f

def circ_transpose_mult_fn(subdiag_f, x):
    # Circular shift
    y = torch.cat((x[1:], x[0]))

    # Scale by [v f]
    return y * subdiag_f

# Tridiagonal + corner operators
# TODO NEEDS FIX
def tridiag_transpose_mult_fn(subdiag_f, diag, supdiag, x):
    y = torch.cat((x[1:], x[0]))
    sub_result = y * subdiag_f
    z = Variable(torch.zeros(1).cuda())
    sup_result = torch.cat((z, x[:-1] * supdiag))
    diag_result = x*diag

    return sup_result + sub_result + diag_result



# Assumes Stein displacement.
def set_mult_fns(self, params):
    # assert params.disp_type == 'stein'
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


