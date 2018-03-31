import functools
import numpy as np

import torch
from torch.autograd import Variable
import pytorch_fft.fft as fft

Rfft = fft.autograd.Rfft()
Irfft = fft.autograd.Irfft()

from triextrafat import krylov_construct

def complex_mult(X, Y):
    X_re, X_im = X
    Y_re, Y_im = Y
    return X_re * Y_re - X_im * Y_im, X_re * Y_im + X_im * Y_re

def krylov_transpose_multiply(subdiag, v, u):
    """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    T_00 = u[:, np.newaxis, ..., np.newaxis] * v[np.newaxis, ..., np.newaxis]
    T_01 = u[..., np.newaxis]
    T_10 = v[..., np.newaxis]
    T_11 = Variable(torch.ones((n, 1))).cuda()
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_00, S_01, S_10, S_11 = T_00, T_01, T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S1_01 = torch.cat((S_01[:, 1::2], torch.zeros_like(S_01[:, 1::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_01, S1_11[np.newaxis]))

        # polynomial multiplications
        S_f_re, S_f_im = Rfft(S)
        S0_10_f_re, S0_11_f_re, S1_01_f_re, S1_11_f_re = S_f_re[:rank], S_f_re[rank], S_f_re[rank+1:rank+1+batch_size], S_f_re[-1]
        S0_10_f_im, S0_11_f_im, S1_01_f_im, S1_11_f_im = S_f_im[:rank], S_f_im[rank], S_f_im[rank+1:rank+1+batch_size], S_f_im[-1]
        S1_01_f = (S1_01_f_re, S1_01_f_im)
        S0_11_f = (S0_11_f_re, S0_11_f_im)
        S1_11_f = (S1_11_f_re, S1_11_f_im)
        S0_10_f = (S0_10_f_re, S0_10_f_im)
        T_00_f_re, T_00_f_im = complex_mult((S1_01_f_re[:, np.newaxis], S1_01_f_im[:, np.newaxis]),
                                            (S0_10_f_re[np.newaxis], S0_10_f_im[np.newaxis]))
        T_01_f_re, T_01_f_im = complex_mult(S1_01_f, S0_11_f)
        T_10_f_re, T_10_f_im = complex_mult(S1_11_f, S0_10_f)
        T_11_f_re, T_11_f_im = complex_mult(S1_11_f, S0_11_f)

        T_f_re = torch.cat((torch.cat((T_00_f_re, T_01_f_re[:, np.newaxis]), dim=1),
                            torch.cat((T_10_f_re[np.newaxis], T_11_f_re[np.newaxis, np.newaxis]), dim=1)))
        T_f_im = torch.cat((torch.cat((T_00_f_im, T_01_f_im[:, np.newaxis]), dim=1),
                            torch.cat((T_10_f_im[np.newaxis], T_11_f_im[np.newaxis, np.newaxis]), dim=1)))

        T = Irfft(T_f_re, T_f_im) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_00, T_01, T_10, T_11 = T[:batch_size, :rank], T[:batch_size, -1], T[-1, :rank], T[-1, -1]

        # polynomial additions
        T_00 = torch.cat((T_00[:, :, :, :n2], T_00[:, :, :, n2:] + S_00[:, :, ::2] + S_00[:, :, 1::2]), dim=-1)
        T_01 = torch.cat((T_01[:, :, :n2], T_01[:, :, n2:] + S_01[:, ::2]), dim=-1)
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    # Negative step isn't supported by Pytorch
    # (https://github.com/pytorch/pytorch/issues/229) so we have to construct
    # the index explicitly.
    reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    return T_00[:, :, :, reverse_index].squeeze(dim=2)

def test():
    m = 12
    n = 1<<m
    batch_size = 512
    rank = 3
    subdiag = Variable(torch.rand(n-1), requires_grad=True).cuda()
    A = np.diag(subdiag.data.cpu().numpy(), -1)
    u = Variable(torch.rand((batch_size, n)), requires_grad=True).cuda()
    v = Variable(torch.rand((rank, n)), requires_grad=True).cuda()
    result1 = krylov_transpose_multiply(subdiag, v, u)
    result1 = result1.data.cpu().numpy()
    # CPU dense multiply
    Ks = [krylov_construct(A, v.data.cpu().numpy()[i], n) for i in range(rank)]
    u_cpu = u.data.cpu().numpy()
    result2 = np.stack([u_cpu @ K.T for K in Ks])
    result2 = result2.swapaxes(0, 1).squeeze()
    # GPU dense multiply
    Ks_pytorch = [Variable(torch.Tensor(K)).cuda() for K in Ks]
    result3 = torch.stack([u @ K.t() for K in Ks_pytorch])
    result3 = result3.data.cpu().numpy().swapaxes(0, 1).squeeze()
    # Explicit construction on GPU
    linear_fn = functools.partial(shift_subdiag, subdiag)
    Ks_gpu = [Krylov(linear_fn, v_) for v_ in v]
    result4 = torch.stack([u @ K for K in Ks_gpu])
    result4 = result4.data.cpu().numpy().swapaxes(0, 1).squeeze()
    # np.allclose(result1, result2)
    print(np.max(abs(result1 - result2)))
    print(np.mean(abs(result1 - result2)))
    print(np.max(abs(result3 - result2)))
    print(np.mean(abs(result3 - result2)))
    print(np.max(abs(result4 - result2)))
    print(np.mean(abs(result4 - result2)))

    a = Variable(torch.rand(3, 4, 8).cuda(), requires_grad=True)
    # b_re, b_im = fft.autograd.Fft(a)
    b_re, b_im = fft.rfft(a.data)

    b = b_re.cpu().numpy() + 1j * b_im.cpu().numpy()
    b_np = np.fft.rfft(a.cpu().numpy())
    np.allclose(b, b_np)

    temp = Variable(torch.zeros_like(a.data))
    temp[0] = a[0]
    s = temp.sum()
    from torch import autograd
    g = autograd.grad(s, a)

def shift(v, f=1):
    return torch.cat((f * v[[v.size(0) - 1]], v[:-1]))

def shift_subdiag(subdiag, v, f=0.0):
    return torch.cat((f * v[[v.size(0) - 1]], subdiag * v[:-1]))

def Krylov(linear_map, v, n=None):
    if n is None:
        n = v.size(0)
    cols = [v]
    for _ in range(n - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)
