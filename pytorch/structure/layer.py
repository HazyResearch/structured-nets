import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from . import toeplitz as toep
from . import krylov as kry
from . import circulant as circ

class StructuredLinear(nn.Module):
    class_type = None
    abbrev = None

    def name(self):
        return self.__class__.abbrev

    def __init__(self, layer_size=None, bias=True, **kwargs):
        super().__init__()
        self.layer_size = layer_size
        self.bias = bias
        self.__dict__.update(kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        assert self.layer_size is not None
        self.b = None
        if self.bias:
            self.b = Parameter(torch.zeros(self.layer_size))

    def apply_bias(self, out):
        if self.b is not None:
            return self.b + out
        else:
            return out

    def loss(self):
        return 0

class Unconstrained(StructuredLinear):
    class_type = 'unconstrained'
    abbrev = 'u'

    def reset_parameters(self):
        super().reset_parameters()
        # TODO: initialize based on stddev automatically
        # self.init_stddev = 0.01
        self.init_stddev = np.sqrt(1./self.layer_size)
        self.W = Parameter(torch.Tensor(self.layer_size, self.layer_size))
        torch.nn.init.normal_(self.W, std=self.init_stddev)

    def forward(self, x):
        out = torch.matmul(x, self.W)
        return self.apply_bias(out)


class Circulant(StructuredLinear):
    class_type = 'circulant'
    abbrev = 'c'

    # def __init__(self, layer_size, init_stddev=0.01, bias=False):
    #     super().__init__()
    #     self.bias =

    def reset_parameters(self):
        super().reset_parameters()
        self.c = Parameter(torch.Tensor(self.layer_size))
        # self.init_stddev = 0.01 # TODO initialize smartly
        self.init_stddev = np.sqrt(1./self.layer_size)
        torch.nn.init.normal_(self.c, std=self.init_stddev)

    def forward(self, x):
        return self.apply_bias(circ.circulant_multiply(self.c, x, self.layer_size))

class LowRank(StructuredLinear):
    class_type = 'low_rank'
    abbrev = 'lr'

    def name(self):
        return self.__class__.abbrev + str(self.r)

    def __init__(self, layer_size, r=1, **kwargs):
        super().__init__(layer_size, r=r, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        self.G = Parameter(torch.Tensor(self.r, self.layer_size))
        self.H = Parameter(torch.Tensor(self.r, self.layer_size))
        # TODO: calculate stddev automatically
        # self.init_stddev = 0.01
        self.init_stddev = np.power(1. / (self.r * self.layer_size), 1/2)
        torch.nn.init.normal_(self.G, std=self.init_stddev)
        torch.nn.init.normal_(self.H, std=self.init_stddev)

    def forward(self, x):
        xH = torch.matmul(x, self.H.t())
        out = torch.matmul(xH, self.G)
        return self.apply_bias(out)


class ToeplitzLike(LowRank):
    class_type = 'toeplitz'
    abbrev = 't'

    # def __init__(self, corner=False, **kwargs):
    #     super().__init__(corner=corner, **kwargs)
    def reset_parameters(self):
        super().reset_parameters()
        self.corner = False

    def forward(self, x):
        out = toep.toeplitz_mult(self.G, self.H, x, self.corner)
        return self.apply_bias(out)

class ToeplitzLikeC(ToeplitzLike):
    class_type = 'toeplitz_corner'
    abbrev = 'tc'

    def reset_parameters(self):
        super().reset_parameters()
        self.corner = True

class HankelLike(LowRank):
    pass
class VandermondeLike(LowRank):
    pass

class LearnedOperator(LowRank):
    """
    Abstract class for learned displacement operators
    Contains parameters such as tie_operators
    """
    class_type = None # abstract
    abbrev = None

    def __init__(self, tie_operators=False, corner=False, **kwargs):
        super().__init__(tie_operators=tie_operators, corner=corner, **kwargs)

class LDRSubdiagonal(LearnedOperator):
    class_type = 'subdiagonal'
    abbrev = 'sd'

    def reset_parameters(self):
        super().reset_parameters()
        self.subd_A = Parameter(torch.ones(self.layer_size-1))
        if self.tie_operators:
            self.subd_B = self.subd_A
        else:
            self.subd_B = Parameter(torch.ones(self.layer_size-1))

    def forward(self, x):
        out = kry.subdiag_mult(self.subd_A, self.subd_B, self.G, self.H, x)
        return self.apply_bias(out)

class LDRSubdiagonalC(LDRSubdiagonal):
    class_type = 'subdiagonal_corner'
    abbrev = 'sdc'

    def reset_parameters(self):
        super().reset_parameters()
        self.corner_A = Parameter(torch.tensor(0.0))
        self.corner_B = Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = kry.subdiag_mult_cuda(self.subd_A, self.subd_B, self.G, self.H, x, corner_A=self.corner_A, corner_B=self.corner_B)
        return self.apply_bias(out)

class LDRTridiagonal(LearnedOperator):
    class_type = 'tridiagonal'
    abbrev = 'td'

    def reset_parameters(self):
        super().reset_parameters()
        self.subd_A = Parameter(torch.ones(self.layer_size-1))
        self.diag_A = Parameter(torch.zeros(self.layer_size))
        self.supd_A = Parameter(torch.zeros(self.layer_size-1))
        if self.tie_operators:
            self.subd_B = self.subd_A
            self.diag_B = self.diag_A
            self.supd_B = self.supd_A
        else:
            self.subd_B = Parameter(torch.ones(self.layer_size-1))
            self.diag_B = Parameter(torch.zeros(self.layer_size))
            self.supd_B = Parameter(torch.zeros(self.layer_size-1))
        self.corners_A = (0.0,0.0)
        self.corners_B = (0.0,0.0)

    def forward(self, x):
        # out = kry.tridiag_mult_slow(self.subd_A, self.diag_A, self.supd_A, self.subd_B, self.diag_B, self.supd_B, self.G, self.H, x)
        out = kry.tridiag_mult_slow(self.subd_A, self.diag_A, self.supd_A, self.subd_B, self.diag_B, self.supd_B, self.G, self.H, x, corners_A=self.corners_A, corners_B=self.corners_B)
        return self.apply_bias(out)

class LDRTridiagonalC(LDRTridiagonal):
    class_type = 'tridiagonal_corner'
    abbrev = 'tdc'

    def reset_parameters(self):
        super().reset_parameters()
        self.corners_A = (Parameter(torch.tensor(0.0)), Parameter(torch.tensor(0.0)))
        self.corners_B = (Parameter(torch.tensor(0.0)), Parameter(torch.tensor(0.0)))

    # def forward(self, x):
    #     out = kry.tridiag_mult_slow(self.subd_A, self.diag_A, self.supd_A, self.subd_B, self.diag_B, self.supd_B, self.G, self.H, x, corners_A=self.corners_A, corners_B=self.corners_B)
    #     return self.apply_bias(out)


# create a map from class names to the Python class
# TODO: should go in some utils file
def descendants(cls):
    """
    Get all subclasses (recursively) of class cls, not including itself
    Assumes no multiple inheritance
    """
    desc = []
    for subcls in cls.__subclasses__():
        desc.append(subcls)
        desc.extend(descendants(subcls))
    return desc

class_map = {}
for cls in descendants(StructuredLinear):
    if cls.class_type is None: continue
    class_map[cls.class_type] = cls
    class_map[cls.abbrev] = cls

    # else:
    #     assert 0, f"{self.__class__.__name__} does not support {self.class_type}"



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

