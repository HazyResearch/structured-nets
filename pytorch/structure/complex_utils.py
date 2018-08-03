''' Utility functions for handling complex tensors: conjugate and complex_mult.
Pytorch (as of 0.4.0) does not support complex tensors, so we store them as
float tensors where the last dimension is 2 (real and imaginary parts).
Cupy does support complex arrays, so we can cast Pytorch tensor to Cupy array
and use the multiplication function there. If Cupy isn't available, we use our
own complex multiplication implemented in Pytorch, which is about 6x slower.
'''

import torch
from torch.utils.dlpack import to_dlpack, from_dlpack


def conjugate(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


def complex_mult(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)
