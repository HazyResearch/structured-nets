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


"""Functions to multiply by an LDR matrix with subdiagonal and tridiagonal
operator matrices.

We implement the fast multiplication for the subdiagonal case.
This comprises two steps: Krylov(g) @ Krylov(h)^T @ u, which are Krylov
transpose multiply and Krylov multiply.

For tridiagonal case, we implement the slow multiplication algorithm: construct
the Krylov matrix then call regular matrix multiply.
"""

import functools
from math import ceil, log2

import diag_mult_cuda

import torch
from torch.nn import functional as F


def krylov_transpose_multiply_conv(subdiag, v, u):
    """Multiply Krylov(A, v_i)^T @ u when A is zero except on the subdiagonal.
    Use either Pytorch's conv1d or FFT for polynomial multiplication, depending
    on polynomial degree. This is the fastest implementation.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, "u and v must have the same last dimension"
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"

    result = torch.zeros((batch_size, rank, n), dtype=u.dtype, device=u.device)
    T_00_sum = u @ v.t()
    result[:, :, 0] += T_00_sum
    T_01 = u[..., None]
    T_10 = v[..., None]
    T_11 = torch.ones(n, device=T_00_sum.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        _, S_01, S_10, S_11 = T_00_sum, T_01, T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1) :: (2 * n2), None]
        # polynomial multiplication
        if n2 <= 128:  # Pick between 2 implementations based on polynomial degree n2
            T_00_sum = F.conv1d(
                S_01[:, 1::2], S0_10_mult_subdiag.flip(2), padding=n2 - 1
            )
        else:
            S = torch.cat(
                (
                    torch.cat((S0_10_mult_subdiag, S_01[:, 1::2])),
                    torch.zeros(
                        (rank + batch_size, n1, n2),
                        dtype=S_10.dtype,
                        device=S_10.device,
                    ),
                ),
                dim=-1,
            )
            # S = torch.cat((S0_10_mult_subdiag, S_01[:, 1::2]))
            S_f = torch.fft.rfft(S)
            S0_10_f, S1_01_f = S_f[:rank], S_f[rank : rank + batch_size]
            T_00_f_sum = (S1_01_f[:, None] * S0_10_f[None]).sum(dim=2)
            T_00_sum = torch.fft.irfft(T_00_f_sum, n=2 * n2)[..., :-1]
        # polynomial additions
        result[:, :, 1 : 2 * n2] += T_00_sum
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1) :: (2 * n2)]
        T_01 = torch.cat(
            (S_01[:, ::2], S_01[:, 1::2] * S0_11_mult_subdiag[:, None]), dim=-1
        )
        T_10 = torch.cat(
            (S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, None]), dim=-1
        )
        T_11 = S0_11_mult_subdiag * S_11[1::2]
    return result


def krylov_transpose_multiply(subdiag, v, u):
    """Multiply Krylov(A, v_i)^T @ u when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, "u and v must have the same last dimension"
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"

    result = torch.zeros((batch_size, rank, n), dtype=u.dtype, device=u.device)
    T_00_sum = u @ v.t()
    result[:, :, 0] = T_00_sum
    T_01 = u[..., None]
    T_10 = v[..., None]
    T_11 = torch.ones(n, device=T_00_sum.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_01, S_10, S_11 = T_01, T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1) :: (2 * n2), None]
        S = torch.cat(
            (
                torch.cat((S0_10_mult_subdiag, S_01[:, 1::2])),
                torch.zeros(
                    (rank + batch_size, n1, n2), dtype=S_10.dtype, device=S_10.device
                ),
            ),
            dim=-1,
        )

        S_f = torch.fft.rfft(S)

        # polynomial multiplications
        S0_10_f, S1_01_f = S_f[:rank], S_f[rank : rank + batch_size]
        T_00_f_sum = (S1_01_f[:, None] * S0_10_f[None]).sum(dim=2)
        T_00_sum = torch.fft.irfft(T_00_f_sum, n=2 * n2)[..., :-1]

        # polynomial additions
        result[:, :, 1 : 2 * n2] += T_00_sum
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1) :: (2 * n2)]
        T_01 = torch.cat(
            (S_01[:, ::2], S_01[:, 1::2] * S0_11_mult_subdiag[:, None]), dim=-1
        )
        T_10 = torch.cat(
            (S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, None]), dim=-1
        )
        T_11 = S0_11_mult_subdiag * S_11[1::2]

    return result


def KTu_traceable(subdiag, v, u):
    """Multiply Krylov(A, v_i)^T @ u when A is zero except on the subdiagonal.
    (WIP) Written to be traceable by Pytorch 1.0 JIT compiler.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    m = int(log2(n))

    T_00_sum = u @ v.t()
    result = T_00_sum.unsqueeze(-1)
    T_01 = u[..., None]
    T_10 = v[..., None]
    T_11 = torch.ones(n, device=T_00_sum.device)
    for d in range(m)[::-1]:
        n2 = 1 << (m - d - 1)
        S_01, S_10, S_11 = T_01, T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1) :: (2 * n2), None]
        S = torch.cat((S0_10_mult_subdiag, S_01[:, 1::2]))
        # polynomial multiplications
        S_f = torch.fft.rfft(S)
        S0_10_f, S1_01_f = S_f[:rank], S_f[rank : rank + batch_size]
        T_00_f_sum = (S1_01_f[:, None] * S0_10_f[None]).sum(dim=2)
        T_00_sum = torch.fft.irfft(T_00_f_sum, n=2 * n2)[..., :-1]

        # polynomial additions
        result = torch.cat(
            (
                result[:, :, :1],
                result[:, :, 1:] + T_00_sum[:, :, : n2 - 1],
                T_00_sum[:, :, n2 - 1 :],
            ),
            dim=-1,
        )
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1) :: (2 * n2)]
        T_01 = torch.cat(
            (S_01[:, ::2], S_01[:, 1::2] * S0_11_mult_subdiag[:, None]), dim=-1
        )
        T_10 = torch.cat(
            (S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, None]), dim=-1
        )
        T_11 = S0_11_mult_subdiag * S_11[1::2]

    return result


def krylov_transpose_multiply_old(subdiag, v, u):
    """Multiply Krylov(A, v_i)^T @ u when A is zero except on the subdiagonal.
    Uses the old algorithm that scales worse when batching.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, "u and v must have the same last dimension"
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"

    T_00 = u[:, None, ..., None] * v[None, ..., None]
    T_01 = u[..., None]
    T_10 = v[..., None]
    T_11 = torch.ones((n, 1), device=T_00.device)
    for d in range(m)[::-1]:
        _, n2 = 1 << d, 1 << (m - d - 1)
        S_00, S_01, S_10, S_11 = T_00, T_01, T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S1_01 = torch.cat((S_01[:, 1::2], torch.zeros_like(S_01[:, 1::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[None], S1_01, S1_11[None]))

        # polynomial multiplications
        S_f = torch.fft.rfft(S, 1)
        # T_f = complex_mult(S_f[rank + 1 :, None], S_f[: rank + 1])
        T_f = S_f[rank + 1 :, None] * S_f[: rank + 1]

        T = torch.fft.irfft(T_f, n=2 * n2) * subdiag[(n2 - 1) :: (2 * n2), None]
        T_00, T_01, T_10, T_11 = (
            T[:batch_size, :rank],
            T[:batch_size, -1],
            T[-1, :rank],
            T[-1, -1],
        )

        # polynomial additions
        T_00 = torch.cat(
            (
                T_00[:, :, :, :n2],
                T_00[:, :, :, n2:] + S_00[:, :, ::2] + S_00[:, :, 1::2],
            ),
            dim=-1,
        )
        T_01 = torch.cat((T_01[:, :, :n2], T_01[:, :, n2:] + S_01[:, ::2]), dim=-1)
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    return T_00.squeeze(dim=2).flip(2)


def krylov_multiply_conv(subdiag, v, w):
    """Multiply sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Since K @ w can be computed by autodiffing K^T @ u, the algorithm is just
    hand-differentiating the code of @krylov_transpose_multiply.
    Use either Pytorch's conv1d or FFT for polynomial multiplication, depending
    on polynomial degree. This is the fastest implementation.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, "w and v must have the same last dimension"
    assert rank == rank_, "w and v must have the same rank"
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"

    # Forward pass. Since K @ w can be computed by autodiffing K^T @ u, we
    # carry out the forward pass K^T @ u for u = 0 here to save the
    # intermediate values. This code is exactly the same as the function
    # @krylov_transpose_multiply, specialized to the case where u = 0.
    save_for_backward = [None] * m
    T_10 = v[..., None]
    T_11 = torch.ones((n), device=T_10.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1) :: (2 * n2), None]
        T_10 = torch.cat(
            (S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, None]), dim=-1
        )
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1) :: (2 * n2)]
        save_for_backward[d] = S0_10_mult_subdiag, S0_11_mult_subdiag
        T_11 = S0_11_mult_subdiag * S_11[1::2]

    # Backward pass
    dT_01 = torch.zeros((batch_size, 1, n), dtype=w.dtype, device=w.device)

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S0_10_mult_subdiag, S0_11_mult_subdiag = save_for_backward[d]
        dS_01 = torch.empty((batch_size, 2 * n1, n2), device=w.device)
        dS_01[:, ::2] = dT_01[:, :, :n2]
        if n2 <= 128:
            dS1_01 = F.conv_transpose1d(
                w[:, :, 1 : 2 * n2], S0_10_mult_subdiag.flip(2), padding=n2 - 1
            )
        else:
            dT_00_sum = torch.cat(
                (
                    w[:, :, 1 : 2 * n2],
                    torch.zeros((batch_size, rank, 1), dtype=w.dtype, device=w.device),
                ),
                dim=-1,
            )
            dT_00_sum_f = torch.fft.rfft(dT_00_sum)
            S0_10_f = torch.fft.rfft(
                torch.cat(
                    (S0_10_mult_subdiag, torch.zeros_like(S0_10_mult_subdiag)), dim=-1
                ),
            )
            dS1_01_f = (S0_10_f.conj() * dT_00_sum_f[:, :, None]).sum(dim=1)
            dS1_01 = torch.fft.irfft(dS1_01_f, n=2 * n2)[:, :, :n2]
        dS_01[:, 1::2] = dT_01[:, :, n2:] * S0_11_mult_subdiag[:, None] + dS1_01

        dT_01 = dS_01

    # du = ((dT_00_sum[:, :, np.newaxis] * v[np.newaxis, :, :, np.newaxis]).sum(dim=1) + dT_01).squeeze(dim=-1)
    du = w[:, :, 0] @ v + dT_01.squeeze(dim=-1)
    return du


def krylov_multiply(subdiag, v, w):
    """Multiply sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Since K @ w can be computed by autodiffing K^T @ u, the algorithm is just
    hand-differentiating the code of @krylov_transpose_multiply.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, "w and v must have the same last dimension"
    assert rank == rank_, "w and v must have the same rank"
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"

    # Forward pass. Since K @ w can be computed by autodiffing K^T @ u, we
    # carry out the forward pass K^T @ u for u = 0 here to save the
    # intermediate values. This code is exactly the same as the function
    # @krylov_transpose_multiply, specialized to the case where u = 0.
    save_for_backward = [None] * m
    T_10 = v[..., None]
    T_11 = torch.ones((n), device=T_10.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1) :: (2 * n2), None]
        T_10 = torch.cat(
            (S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, None]), dim=-1
        )
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1) :: (2 * n2)]
        save_for_backward[d] = S0_10_mult_subdiag, S0_11_mult_subdiag
        T_11 = S0_11_mult_subdiag * S_11[1::2]

    # Backward pass
    dT_01 = torch.zeros((batch_size, 1, n), dtype=w.dtype, device=w.device)

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S0_10_mult_subdiag, S0_11_mult_subdiag = save_for_backward[d]
        dS_01 = torch.empty((batch_size, 2 * n1, n2), device=w.device)
        dS_01[:, ::2] = dT_01[:, :, :n2]
        dT_00_sum = torch.cat(
            (
                w[:, :, 1 : 2 * n2],
                torch.zeros((batch_size, rank, 1), dtype=w.dtype, device=w.device),
            ),
            dim=-1,
        )

        dT_00_sum_f = torch.fft.rfft(dT_00_sum)
        S0_10_f = torch.fft.rfft(
            torch.cat(
                (S0_10_mult_subdiag, torch.zeros_like(S0_10_mult_subdiag)), dim=-1
            )
        )
        dS1_01_f = (S0_10_f.conj() * dT_00_sum_f[:, :, None]).sum(dim=1)
        dS1_01 = torch.fft.irfft(dS1_01_f, n=2 * n2)[:, :, :n2]
        dS_01[:, 1::2] = dT_01[:, :, n2:] * S0_11_mult_subdiag[:, None] + dS1_01

        dT_01 = dS_01

    # du = ((dT_00_sum[:, :, np.newaxis] * v[np.newaxis, :, :, np.newaxis]).sum(dim=1) + dT_01).squeeze(dim=-1)
    du = w[:, :, 0] @ v + dT_01.squeeze(dim=-1)
    return du


def krylov_multiply_by_autodiff(subdiag, v, w):
    """Multiply sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal, using Pytorch's autodiff.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, "w and v must have the same last dimension"
    assert rank == rank_, "w and v must have the same rank"
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"

    u = torch.zeros((batch_size, n), dtype=v.dtype, device=v.device, requires_grad=True)
    prod = krylov_transpose_multiply(subdiag, v, u)
    (result,) = torch.autograd.grad(prod, u, grad_outputs=w, create_graph=True)
    return result


def krylov_multiply_forward_old_(subdiag, v):
    """Forward pass of Krylov_multiply. Since K @ w can be computed by
    autodiffing K^T @ u, we carry out the forward pass K^T @ u for u = 0 here
    to save the intermediate values. This code is exactly the same as the
    function @krylov_transpose_multiply_old, specialized to the case where u = 0.
    Uses the old algorithm that scales worse when batching.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
    Returns:
        save_for_backward: list of length log n, containing intermediate values
    necessary for the backward pass K @ w.
    """
    rank, n = v.shape
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"

    save_for_backward = [None] * m
    T_10 = v[..., None]
    T_11 = torch.ones((n, 1), device=T_10.device)
    for d in range(m)[::-1]:
        _, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[None], S1_11[None]))

        # polynomial multiplications
        S_f = torch.fft.rfft(S, 1)
        save_for_backward[d] = S_f[: rank + 1]
        # T_f = complex_mult(S_f[-1], S_f[: rank + 1])
        T_f = S_f[-1] * S_f[: rank + 1]

        T = (
            torch.irfft(T_f, 1, signal_sizes=(2 * n2,))
            * subdiag[(n2 - 1) :: (2 * n2), None]
        )
        T_10, T_11 = T[:rank], T[-1]

        # polynomial additions
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    return save_for_backward


def krylov_multiply_old(subdiag, v, w):
    """Multiply sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Since K @ w can be computed by autodiffing K^T @ u, the algorithm is just
    hand-differentiating the code of @krylov_transpose_multiply.
    Uses the old algorithm that scales worse when batching.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, "w and v must have the same last dimension"
    assert rank == rank_, "w and v must have the same rank"
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"

    save_for_backward = krylov_multiply_forward_old_(subdiag, v)
    w = w[:, :, None, :]
    dT_00, dT_01 = w.flip(w.dim() - 1), torch.zeros(
        (batch_size, 1, n), dtype=w.dtype, device=w.device
    )

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        dS_00 = torch.empty((batch_size, rank, 2 * n1, n2), device=w.device)
        dS_00[:, :, ::2] = dT_00[:, :, :, n2:]
        dS_00[:, :, 1::2] = dT_00[:, :, :, n2:]
        dS_01 = torch.empty((batch_size, 2 * n1, n2), device=w.device)
        dS_01[:, ::2] = dT_01[:, :, n2:]

        dT = torch.cat((dT_00, dT_01[:, None]), dim=1)
        dT = dT * subdiag[(n2 - 1) :: (2 * n2), None]

        dT_f = torch.fft.rfft(dT, 1) / (2 * n2)

        dS1_01_f = (save_for_backward[d].conj() * dT_f).sum(dim=1)

        dS1_01 = torch.irfft(dS1_01_f, 1, signal_sizes=(2 * n2,)) * (2 * n2)
        dS_01[:, 1::2] = dS1_01[:, :, :n2]

        dT_00, dT_01 = dS_00, dS_01

    du = ((dT_00 * v[None, :, :, None]).sum(dim=1) + dT_01).squeeze(dim=-1)
    return du


def subdiag_mult_conv(subdiag_A, subdiag_B, G, H, x):
    """Multiply sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x when A and B are zero except on the subdiagonal.
    Uses the fast algorithm.
    Use either Pytorch's conv1d or FFT for polynomial multiplication, depending
    on polynomial degree. This is the fastest implementation.
    Parameters:
        subdiag_A: Tensor of shape (n - 1, )
        subdiag_B: Tensor of shape (n - 1, )
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    rank, n = G.shape
    batch_size = x.shape[0]
    # if not power of 2, round everything up
    # TODO: this can maybe be handled better. also should benchmark how much speed non-po2 FFT loses
    m = int(ceil(log2(n)))
    n_extended = 1 << m
    if n != n_extended:
        x = torch.cat(
            (
                x,
                torch.zeros(batch_size, n_extended - n, dtype=x.dtype, device=x.device),
            ),
            dim=-1,
        )
        G = torch.cat(
            (G, torch.zeros(rank, n_extended - n, dtype=G.dtype, device=G.device)),
            dim=-1,
        )
        H = torch.cat(
            (H, torch.zeros(rank, n_extended - n, dtype=H.dtype, device=H.device)),
            dim=-1,
        )
        subdiag_A = torch.cat(
            (
                subdiag_A,
                torch.zeros(
                    n_extended - n, dtype=subdiag_A.dtype, device=subdiag_A.device
                ),
            )
        )
        subdiag_B = torch.cat(
            (
                subdiag_B,
                torch.zeros(
                    n_extended - n, dtype=subdiag_B.dtype, device=subdiag_B.device
                ),
            )
        )
    KT_out = krylov_transpose_multiply_conv(subdiag_B, H, x)
    K_out = krylov_multiply_conv(subdiag_A, G, KT_out)
    return K_out[:, :n] if n != n_extended else K_out


def subdiag_mult(subdiag_A, subdiag_B, G, H, x):
    """Multiply sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x when A and B are zero except on the subdiagonal.
    Uses the fast algorithm.
    Parameters:
        subdiag_A: Tensor of shape (n - 1, )
        subdiag_B: Tensor of shape (n - 1, )
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    rank, n = G.shape
    batch_size = x.shape[0]
    # if not power of 2, round everything up
    # TODO: this can maybe be handled better. also should benchmark how much speed non-po2 FFT loses
    m = int(ceil(log2(n)))
    n_extended = 1 << m
    if n != n_extended:
        x = torch.cat(
            (
                x,
                torch.zeros(batch_size, n_extended - n, dtype=x.dtype, device=x.device),
            ),
            dim=-1,
        )
        G = torch.cat(
            (G, torch.zeros(rank, n_extended - n, dtype=G.dtype, device=G.device)),
            dim=-1,
        )
        H = torch.cat(
            (H, torch.zeros(rank, n_extended - n, dtype=H.dtype, device=H.device)),
            dim=-1,
        )
        subdiag_A = torch.cat(
            (
                subdiag_A,
                torch.zeros(
                    n_extended - n, dtype=subdiag_A.dtype, device=subdiag_A.device
                ),
            )
        )
        subdiag_B = torch.cat(
            (
                subdiag_B,
                torch.zeros(
                    n_extended - n, dtype=subdiag_B.dtype, device=subdiag_B.device
                ),
            )
        )
    KT_out = krylov_transpose_multiply(subdiag_B, H, x)
    K_out = krylov_multiply(subdiag_A, G, KT_out)
    return K_out[:, :n] if n != n_extended else K_out


##### Slow multiplication for the subdiagonal case


def Krylov(linear_map, v, m=None):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{m-1} @ v].
    Parameters:
        linear_map: a function v -> A @ v that takes a vector of size m and returns a vector of size m.
        v: the starting vector of size m or (rank, m).
        m: max power of A.
    Returns:
        K: Krylov matrix of size (m, m) or (rank, m, m).
    """
    if m is None:
        m = v.size(-1)
    cols = [v]
    for _ in range(m - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)


def shift_subdiag(subdiag, v, upper_right_corner=0.0):
    """The linear map for multiplying with a subdiagonal matrix (possibly with an upper right corner).
    This implementation is slow and not batched wrt rank, but easy to understand.
    Parameters:
        subdiag: (n - 1, )
        v: (n, )
        upper_right_corner: real number
    Returns:
        prod: (n, )
    """
    return torch.cat((upper_right_corner * v[[-1]], subdiag * v[:-1]))


def subdiag_linear_map(subdiag, upper_right_corner=0.0):
    """Construct the linear map for multiplying with a subdiagonal matrix (possibly with an upper right corner).
    This implementation is faster. The slowness of the Krylov construction is
    from the kernel launch overhead in CUDA: we have n sequential steps, each
    step having a few CUDA calls. To make it faster, we want to reduce the
    number of CUDA operations. Here we reduce each step to 2 operations:
    indexing, and pointwise multiplication.
    Parameters:
        subdiag: (n - 1, )
        upper_right_corner: real number
    Returns:
        linear_map: v -> product, with v of shape either (n, ) or (rank, n)
    """
    n = subdiag.size(0) + 1
    shift_down = torch.arange(-1, n - 1, device=subdiag.device)
    subdiag_extended = torch.cat(
        (
            torch.tensor(
                [upper_right_corner], dtype=subdiag.dtype, device=subdiag.device
            ),
            subdiag,
        )
    )
    # Pytorch 1.0 has torch.roll that should be much faster
    # return lambda v: subdiag_extended * v.roll(1, dims=-1)
    return lambda v: subdiag_extended * v[..., shift_down]


def krylov_subdiag_fast(subdiag, v, upper_right_corner=0.0):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{n-1} @ v]
    where A is a subdiagonal matrix (possibly with an upper right corner).
    This uses vectorized indexing and cumprod so it's much faster than using
    the Krylov function. However, the backward pass is slow because of
    inefficient implementation of cumprod_backward in Pytorch.
    This should yields similar speed (forward + backward) to the fast
    multiplication algorithm, but requires more memory.
    Parameters:
        subdiag: (n - 1, )
        v: the starting vector of size n or (rank, n).
        upper_right_corner: real number
    Returns:
        K: Krylov matrix of size (n, n) or (rank, n, n).
    """
    rank, n = v.shape
    a = torch.arange(n, dtype=torch.long, device=v.device)
    b = -a
    indices = a[:, None] + b[None]
    v_circulant = v[:, indices]
    subdiag_extended = torch.cat(
        (
            torch.tensor(
                [upper_right_corner], dtype=subdiag.dtype, device=subdiag.device
            ),
            subdiag,
        )
    )
    subdiag_circulant = subdiag_extended[indices]
    subdiag_cumprod = subdiag_circulant.cumprod(dim=1)
    K = v_circulant
    K[:, :, 1:] *= subdiag_cumprod[:, :-1]
    return K


def subdiag_mult_slow_old(subdiag_A, subdiag_B, G, H, x):
    """Multiply sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x when A and B are zero except on the subdiagonal.
    Uses the explicit Krylov construction with slow (and easy to understand)
    linear map.
    Parameters:
        subdiag_A: Tensor of shape (n - 1, )
        subdiag_B: Tensor of shape (n - 1, )
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    rank, n = G.shape
    linear_map_A = functools.partial(shift_subdiag, subdiag_A)
    linear_map_B = functools.partial(shift_subdiag, subdiag_B)
    krylovs = [
        (Krylov(linear_map_A, G[i]), Krylov(linear_map_B, H[i]).t())
        for i in range(rank)
    ]
    prods = [K[0] @ (K[1] @ x.t()) for K in krylovs]
    return sum(prods).t()


def subdiag_mult_slow(subdiag_A, subdiag_B, G, H, x, corner_A=0.0, corner_B=0.0):
    """Multiply sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x when A and B are zero except on the subdiagonal.
    Uses the explicit Krylov construction with the more careful implementation of linear map.
    Parameters:
        subdiag_A: Tensor of shape (n - 1, )
        subdiag_B: Tensor of shape (n - 1, )
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    if G.shape[0] == 1:  # specialized code for rank=1, giving 2x speedup.
        K_G = Krylov(subdiag_linear_map(subdiag_A, corner_A), G[0])
        K_H = Krylov(subdiag_linear_map(subdiag_B, corner_B), H[0])
        return (x @ K_H) @ K_G.t()
    else:
        K_G = Krylov(subdiag_linear_map(subdiag_A, corner_A), G)
        K_H = Krylov(subdiag_linear_map(subdiag_B, corner_B), H)
        return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)


def subdiag_mult_slow_fast(subdiag_A, subdiag_B, G, H, x):
    """Multiply sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x when A and B are zero except on the subdiagonal.
    Uses the fast construction of Krylov matrix.
    Parameters:
        subdiag_A: Tensor of shape (n - 1, )
        subdiag_B: Tensor of shape (n - 1, )
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    K_G, K_H = krylov_subdiag_fast(subdiag_A, G), krylov_subdiag_fast(subdiag_B, H)
    return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)


class CycleDownMultCuda(torch.autograd.Function):
    """Cycle v down and do pointwise multiplication with subdiag."""

    @staticmethod
    def forward(ctx, subdiag, v):
        ctx.save_for_backward(subdiag, v)
        return diag_mult_cuda.cycle_mult(subdiag, v, 0, -1)

    @staticmethod
    def backward(ctx, grad):
        subdiag, v = ctx.saved_tensors
        return diag_mult_cuda.cycle_mult(grad, v, 0, -1).sum(
            dim=0
        ), diag_mult_cuda.cycle_mult(subdiag, grad, 1, 1)


cycle_down_mult = CycleDownMultCuda.apply


def subdiag_linear_map_cuda(subdiag, upper_right_corner=0.0):
    """Construct the linear map for multiplying with a subdiagonal matrix (possibly with an upper right corner).
    Uses the construction in CUDA, so it's pretty fast.
    Parameters:
        subdiag: (n - 1, )
        upper_right_corner: real number
    Returns:
        linear_map: v -> product, with v of shape either (n, ) or (rank, n)
    """
    subdiag_extended = torch.cat(
        (
            torch.tensor(
                [upper_right_corner], dtype=subdiag.dtype, device=subdiag.device
            ),
            subdiag,
        )
    )
    return lambda v: cycle_down_mult(subdiag_extended, v)


def subdiag_mult_cuda(subdiag_A, subdiag_B, G, H, x, corner_A=0.0, corner_B=0.0):
    """Multiply sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x when A and B are zero except on the subdiagonal.
    Uses the explicit Krylov construction in CUDA.
    Parameters:
        subdiag_A: Tensor of shape (n - 1, )
        subdiag_B: Tensor of shape (n - 1, )
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    K_G = Krylov(subdiag_linear_map_cuda(subdiag_A, corner_A), G)
    K_H = Krylov(subdiag_linear_map_cuda(subdiag_B, corner_B), H)
    return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)


##### Slow multiplication for the tridiagonal case


def tridiag_linear_map(
    subdiag, diag, superdiag, upper_right_corner=0.0, lower_left_corner=0.0
):
    """Construct the linear map for multiplying with a tridiagonal matrix
    (possibly with upper right and lower left corners).
    Similar to subdiag_linear_map, we want to reduce the number of CUDA
    operations. Here we reduce each step to 3 operations: indexing,
    pointwise multiplication, and summing.
    Parameters:
        subdiag: (n - 1, )
        diag: (n, )
        superdiag: (n - 1, )
        upper_right_corner: real number
        lower_left_corner: real number
    Returns:
        linear_map: v -> product, with v of shape either (n, ) or (rank, n)
    """
    n = diag.size(0)
    shift_none = torch.arange(n, device=diag.device)
    shift_down = shift_none - 1
    shift_up = (shift_none + 1) % n
    shifts = torch.stack((shift_down, shift_none, shift_up))
    subdiag_extended = torch.cat(
        (
            torch.tensor(
                [upper_right_corner], dtype=subdiag.dtype, device=subdiag.device
            ),
            subdiag,
        )
    )
    superdiag_extended = torch.cat(
        (
            superdiag,
            torch.tensor(
                [lower_left_corner], dtype=superdiag.dtype, device=superdiag.device
            ),
        )
    )
    diags = torch.stack((subdiag_extended, diag, superdiag_extended))
    return lambda v: (diags * v[..., shifts]).sum(dim=-2)


def tridiag_linear_map_slow(
    subdiag, diag, superdiag, upper_right_corner=0.0, lower_left_corner=0.0
):
    """The linear map for multiplying with a tridiagonal matrix (possibly with
    upper right and lower left corner).
    This implementation is slow, but easy to understand.
    Parameters:
        subdiag: (n - 1, )
        diag: (n, )
        superdiag: (n - 1, )
        upper_right_corner: real number
        lower_left_corner: real number
    Returns:
        linear_map: v -> product, with v of shape either (n, ) or (rank, n)
    """
    return (
        lambda v: torch.cat(
            (upper_right_corner * v[..., -1:], subdiag * v[..., :-1]), dim=-1
        )
        + diag * v
        + torch.cat((superdiag * v[..., 1:], lower_left_corner * v[..., :1]), dim=-1)
    )


def tridiag_mult_slow(
    subdiag_A,
    diag_A,
    superdiag_A,
    subdiag_B,
    diag_B,
    superdiag_B,
    G,
    H,
    x,
    corners_A=(0.0, 0.0),
    corners_B=(0.0, 0.0),
):
    """Multiply sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x when A and B are zero except on the subdiagonal.
    Uses the explicit Krylov construction with the more careful implementation of linear map.
    Parameters:
        subdiag_A: Tensor of shape (n - 1, )
        diag_A: Tensor of shape (n, )
        superdiag_A: Tensor of shape (n - 1, )
        subdiag_B: Tensor of shape (n - 1, )
        diag_B: Tensor of shape (n, )
        superdiag_B: Tensor of shape (n - 1, )
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        corners_A: two real numbers, the upper right and lower left corners of A.
        corners_B: two real numbers, the upper right and lower left corners of A.
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    if G.shape[0] == 1:  # specialized code for rank=1, giving 2x speedup.
        K_G = Krylov(
            tridiag_linear_map(subdiag_A, diag_A, superdiag_A, *corners_A), G[0]
        )
        K_H = Krylov(
            tridiag_linear_map(subdiag_B, diag_B, superdiag_B, *corners_B), H[0]
        )
        return (x @ K_H) @ K_G.t()
    else:
        K_G = Krylov(tridiag_linear_map(subdiag_A, diag_A, superdiag_A, *corners_A), G)
        K_H = Krylov(tridiag_linear_map(subdiag_B, diag_B, superdiag_B, *corners_B), H)
        return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)
