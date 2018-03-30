from triextrafat import *
import numpy as np

import torch
from torch.autograd import Variable
import pytorch_fft.fft as fft

Rfft = fft.autograd.Rfft()
Irfft = fft.autograd.Irfft()

def complex_mult(X, Y):
    X_re, X_im = X
    Y_re, Y_im = Y
    return (X_re * Y_re - X_im * Y_im, X_re * Y_im + X_im * Y_re)


def krylov_transpose_multiply(subdiag, v, u):
    """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    # u, v = u.view(batch_size, 1, n, 1), v.view(1, rank, n, 1)
    T_00_prev = u[:, np.newaxis, ..., np.newaxis] * v[np.newaxis, ..., np.newaxis]
    T_01_prev = u[..., np.newaxis]
    T_10_prev = v[..., np.newaxis]
    T_11_prev = Variable(torch.ones((n, 1))).cuda()
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_00, S_01, S_10, S_11 = T_00_prev, T_01_prev, T_10_prev, T_11_prev
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=2)
        S1_01 = torch.cat((S_01[:, 1::2], torch.zeros_like(S_01[:, 1::2])), dim=2)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=1)

        # polynomial multiplications
        S0_10_f, S0_11_f, S1_01_f, S1_11_f = Rfft(S0_10), Rfft(S0_11), Rfft(S1_01), Rfft(S1_11)
        S1_01_f_re, S1_01_f_im = S1_01_f
        S0_10_f_re, S0_10_f_im = S0_10_f
        T_00_f = complex_mult((S1_01_f_re[:, np.newaxis], S1_01_f_im[:, np.newaxis]), (S0_10_f_re[np.newaxis], S0_10_f_im[np.newaxis]))
        T_01_f = complex_mult(S1_01_f, S0_11_f)
        T_10_f = complex_mult(S1_11_f, S0_10_f)
        T_11_f = complex_mult(S1_11_f, S0_11_f)

        T_00, T_01, T_10, T_11 = Irfft(*T_00_f), Irfft(*T_01_f), Irfft(*T_10_f), Irfft(*T_11_f)
        T_00 = T_00 * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_01 = T_01 * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_10 = T_10 * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_11 = T_11 * subdiag[(n2 - 1)::(2 * n2), np.newaxis]

        # polynomial additions
        T_00 = torch.cat((T_00[:, :, :, :n2], T_00[:, :, :, n2:] + S_00[:, :, ::2] + S_00[:, :, 1::2]), dim=-1)
        T_01 = torch.cat((T_01[:, :, :n2], T_01[:, :, n2:] + S_01[:, ::2]), dim=-1)
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

        T_00_prev, T_01_prev, T_10_prev, T_11_prev = T_00, T_01, T_10, T_11

    index = torch.arange(n - 1, -1, -1).long().cuda()
    # return T_00[:, :, :, ::-1]
    return T_00[:, :, :, index].squeeze(dim=2)

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
    Ks = [krylov_construct(A, v.data.cpu().numpy()[i], n) for i in range(rank)]
    result2 = np.stack([u.data.cpu().numpy() @ K.T for K in Ks]).swapaxes(0, 1).squeeze()
    # np.allclose(result1.data.cpu().numpy(), result2)
    print(np.max(abs(result1.data.cpu().numpy() - result2)))
    print(np.mean(abs(result1.data.cpu().numpy() - result2)))

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
