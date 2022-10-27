# Copyright 2018 HazyResearch
# https://github.com/HazyResearch/structured-nets
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements.
E.g.
[1 2 3 4]
[5 1 2 3]
[9 5 1 2]
"""

import torch

from .krylov import Krylov


def toeplitz_krylov_transpose_multiply(v, u, f=0.0):
    """Multiply Krylov(Z_f, v_i)^T @ u.
    Parameters:
        v: (rank, n)
        u: (batch_size, n)
        f: real number
    Returns:
        product: (batch, rank, n)
    """
    _, n = u.shape
    _, n_ = v.shape
    assert n == n_, "u and v must have the same last dimension"
    if f != 0.0:  # cycle version
        eta = torch.tensor(f, dtype=torch.complex64) ** (
            torch.arange(n, dtype=u.dtype, device=u.device) / n
        )
        u_f = torch.fft.ifft(1 / eta * u)
        v_f = torch.fft.fft(eta * v)
        uv = torch.fft.fft(u_f[:, None] * v_f[None])
        return (eta * uv).real
    else:
        u_f = torch.fft.rfft(torch.cat((u.flip(1), torch.zeros_like(u)), dim=-1))
        v_f = torch.fft.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1))
        uv_f = u_f[:, None] * v_f[None]
        return torch.fft.irfft(uv_f)[..., :n].flip(2)


def toeplitz_krylov_multiply(v, w, f=0.0):
    """Multiply sum_i Krylov(Z_f, v_i) @ w_i.
    Parameters:
        v: (rank, n)
        w: (batch_size, rank, n)
        f: real number
    Returns:
        product: (batch, n)
    """
    _, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, "w and v must have the same last dimension"
    assert rank == rank_, "w and v must have the same rank"
    if f != 0.0:  # cycle version
        eta = torch.tensor(f, dtype=torch.complex64) ** (
            torch.arange(n, dtype=v.dtype, device=v.device) / n
        )
        w_f = torch.fft.fft(1 / eta * w)
        v_f = torch.fft.fft(eta * v)
        wv_sum_f = (w_f * v_f).sum(dim=1)  # Does this happen in the right space?
        wv_sum = torch.fft.ifft(wv_sum_f, 1)
        return (1 / eta * wv_sum).real
    else:
        w_f = torch.fft.rfft(torch.cat((w, torch.zeros_like(w)), dim=-1))
        v_f = torch.fft.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1))
        wv_sum_f = (w_f * v_f).sum(dim=1)
        # return torch.fft.irfft(wv_sum_f, 1, signal_sizes=(2 * n,))[..., :n]
        return torch.fft.irfft(wv_sum_f)[..., :n]


def toeplitz_krylov_multiply_by_autodiff(v, w, f=0.0):
    """Multiply sum_i Krylov(Z_f, v_i) @ w_i, using Pytorch's autodiff.
    This function is just to check the result of toeplitz_krylov_multiply.
    Parameters:
        v: (rank, n)
        w: (batch_size, rank, n)
        f: real number
    Returns:
        product: (batch, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, "w and v must have the same last dimension"
    assert rank == rank_, "w and v must have the same rank"

    u = torch.zeros((batch_size, n), dtype=v.dtype, device=v.device, requires_grad=True)
    prod = toeplitz_krylov_transpose_multiply(v, u, f)
    (result,) = torch.autograd.grad(prod, u, grad_outputs=w, create_graph=True)
    return result


def toeplitz_mult(G, H, x, cycle=True):
    """Multiply sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Parameters:
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    f = (1, -1) if cycle else (0, 0)
    transpose_out = toeplitz_krylov_transpose_multiply(H, x, f[1])
    return toeplitz_krylov_multiply(G, transpose_out, f[0])


##### Slow multiplication for the Toeplitz-like case


def toeplitz_Z_f_linear_map(f=0.0):
    """The linear map for multiplying by Z_f.
    This implementation is slow and not batched wrt rank, but easy to understand.
    Parameters:
        f: real number
    Returns:
        linear_map: v -> product, with v of shape (n, )
    """
    return lambda v: torch.cat((f * v[[-1]], v[:-1]))


def krylov_toeplitz_fast(v, f=0.0):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{n-1} @ v]
    where A = Z_f. This uses vectorized indexing and cumprod so it's much
    faster than using the Krylov function.
    Parameters:
        v: the starting vector of size n or (rank, n).
        f: real number
    Returns:
        K: Krylov matrix of size (n, n) or (rank, n, n).
    """
    rank, n = v.shape
    a = torch.arange(n, device=v.device)
    b = -a
    indices = a[:, None] + b[None]
    K = v[:, indices]
    K[:, indices < 0] *= f
    return K


def toeplitz_mult_slow(G, H, x, cycle=True):
    """Multiply sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Uses the explicit Krylov construction with slow (and easy to understand)
    linear map.
    Parameters:
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    assert G.shape == H.shape, "G and H must have the same shape"
    rank, n = G.shape
    f = (1, -1) if cycle else (0, 0)
    krylovs = [
        (
            Krylov(toeplitz_Z_f_linear_map(f[0]), G[i]),
            Krylov(toeplitz_Z_f_linear_map(f[1]), H[i]).t(),
        )
        for i in range(rank)
    ]
    prods = [K[0] @ (K[1] @ x.t()) for K in krylovs]
    return sum(prods).t()


def toeplitz_mult_slow_fast(G, H, x, cycle=True):
    """Multiply sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Uses the fast construction of Krylov matrix.
    Parameters:
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    assert G.shape == H.shape
    f_G, f_H = (1, -1) if cycle else (0, 0)
    K_G, K_H = krylov_toeplitz_fast(G, f_G), krylov_toeplitz_fast(H, f_H)
    return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)
