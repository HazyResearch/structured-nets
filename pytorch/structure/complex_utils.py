''' Utility functions for handling complex tensors: conjugate and complex_mult.
Pytorch (as of 0.4.0) does not support complex tensors, so we store them as
float tensors where the last dimension is 2 (real and imaginary parts).
'''

import torch


def conjugate(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


def complex_mult(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    # prod = X * Y
    # out = torch.empty_like(prod)
    # out[..., 0] = prod[..., 0] - prod[..., 1]
    # out[..., 1] = X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]
    # return out
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)
