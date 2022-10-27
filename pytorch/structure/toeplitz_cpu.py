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

from math import log2

import torch


class KT_Toeplitz:
    """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal."""

    def __init__(self, n, f=0, batch_size=1, rank=1):
        m = int(log2(n))
        assert n == 1 << m, "n must be a power of 2"
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.rank = rank

        self.eta = f ** (torch.arange(n).to(torch.complex64) / n) if f != 0 else None

    def __call__(self, v, u):
        """
        Multiply Krylov(Z_f, v)^T @ u
        v: (rank, n)
        u: (batch, n)
        out: (batch, rank, n)
        """
        n, _, batch_size, rank = self.n, self.m, self.batch_size, self.rank

        if self.eta is not None:  # cycle version
            eta = self.eta.to(u.device)
            u_ = torch.fft.ifft(1 / eta * u)
            v_ = torch.fft.fft(eta * v)
            # uv_ = u_.reshape(batch_size, 1, n) * v_.reshape(1, rank, n)
            # ans = eta * torch.fft.fft(uv_)
            # return torch.real(ans)
            uv = torch.fft.fft(u_[:, None] * v_[None])
            return (eta * uv).real
        else:
            u_ = torch.fft.rfft(
                torch.concatenate((u.flip(-1), torch.zeros_like(u)), dim=-1)
            )
            v_ = torch.fft.rfft(torch.concatenate((v, torch.zeros_like(v)), dim=-1))
            uv_ = u_.reshape(batch_size, 1, -1) * v_.reshape(1, rank, -1)
            # ans = torch.fft.irfft(uv_)[..., n - 1 :: -1]
            ans = torch.fft.irfft(uv_)[..., :n].flip(-1)
            return ans


class K_Toeplitz:
    """Multiply Krylov(A, v) @ w when A is zero except on the subdiagonal."""

    def __init__(self, n, f, batch_size=1, rank=1):
        m = int(log2(n))
        assert n == 1 << m, "n must be a power of 2"
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.rank = rank

        self.eta = f ** (torch.arange(n).to(torch.complex64) / n) if f != 0 else None

    def __call__(self, v, w):
        """
        v: (rank, n)
        w: (batch_size, rank, n)
        out: (batch_size, n)
        """
        n, _, _, rank = self.n, self.m, self.batch_size, self.rank
        if self.eta is not None:
            eta = self.eta.to(v.device)
            w_ = torch.fft.fft(eta * w)
            v_ = torch.fft.fft(eta * v)
            wv_ = w_ * v_.reshape((1, rank, n))
            ans = 1 / eta * torch.fft.ifft(torch.sum(wv_, dim=1))
            ans = torch.real(ans)
        else:
            w_ = torch.fft.rfft(torch.concatenate((w, torch.zeros_like(w)), dim=-1))
            v_ = torch.fft.rfft(torch.concatenate((v, torch.zeros_like(v)), dim=-1))
            wv_ = w_ * v_.reshape((1, rank, -1))
            ans = torch.fft.irfft(torch.sum(wv_, dim=1))[..., :n]
        return ans


def toeplitz_mult(G, H, x, cycle=True):
    rank, n = G.shape
    batch_size = x.shape[0]
    f = (1, -1) if cycle else (0, 0)
    transpose_out = KT_Toeplitz(n, f[1], batch_size, rank)(H, x)
    krylov_out = K_Toeplitz(n, f[0], batch_size, rank)(G, transpose_out)
    return krylov_out


##### Slow mult


def krylov_construct(f, v, m):
    n = v.shape[0]
    K = torch.zeros(size=(m, n), device=f.device)
    K[0, :] = v
    for i in range(1, m):
        K[i, 1:] = K[i - 1, :-1]
        K[i, 0] = f * K[i - 1, -1]
    return K.T


def toeplitz_mult_slow(G, H, x, cycle=True):
    assert G.shape == H.shape
    rank, n = G.shape
    f = (1, -1) if cycle else (0, 0)
    krylovs = [
        (krylov_construct(f[0], G[i], n), krylov_construct(f[1], H[i], n).T)
        for i in range(rank)
    ]
    prods = [K[0] @ K[1] @ x.T for K in krylovs]
    return torch.sum(torch.tensor(prods), dim=0).T
