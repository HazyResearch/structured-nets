import functools
import numpy as np

import torch
from torch.autograd import Variable

import cufat as cf
from triextrafat import krylov_construct
# from triXXF import KrylovTransposeMultiply


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
    assert n == n_, 'u and v must have the same last dimension'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    T_00 = u[:, np.newaxis, ..., np.newaxis] * v[np.newaxis, ..., np.newaxis]
    T_01 = u[..., np.newaxis]
    T_10 = v[..., np.newaxis]
    # T_11 = Variable(torch.ones((n, 1))).cuda()
    T_11 = Variable(torch.cuda.FloatTensor(n, 1).fill_(1.0))
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_00, S_01, S_10, S_11 = T_00, T_01, T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S1_01 = torch.cat((S_01[:, 1::2], torch.zeros_like(S_01[:, 1::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_01, S1_11[np.newaxis]))

        # polynomial multiplications
        S_f_re, S_f_im = cf.Rfft_slow()(S)
        S0_10_f_re, S0_11_f_re, S1_01_f_re, S1_11_f_re = S_f_re[:rank], S_f_re[rank], S_f_re[rank+1:rank+1+batch_size], S_f_re[-1]
        S0_10_f_im, S0_11_f_im, S1_01_f_im, S1_11_f_im = S_f_im[:rank], S_f_im[rank], S_f_im[rank+1:rank+1+batch_size], S_f_im[-1]
        S1_01_f = (S1_01_f_re, S1_01_f_im)
        S0_11_f = (S0_11_f_re, S0_11_f_im)
        S1_11_f = (S1_11_f_re, S1_11_f_im)
        S0_10_f = (S0_10_f_re, S0_10_f_im)
        T_00_f_re, T_00_f_im = cf.complex_mult((S1_01_f_re[:, np.newaxis], S1_01_f_im[:, np.newaxis]),
                                            (S0_10_f_re[np.newaxis], S0_10_f_im[np.newaxis]))
        T_01_f_re, T_01_f_im = cf.complex_mult(S1_01_f, S0_11_f)
        T_10_f_re, T_10_f_im = cf.complex_mult(S1_11_f, S0_10_f)
        T_11_f_re, T_11_f_im = cf.complex_mult(S1_11_f, S0_11_f)

        T_f_re = torch.cat((torch.cat((T_00_f_re, T_01_f_re[:, np.newaxis]), dim=1),
                            torch.cat((T_10_f_re[np.newaxis], T_11_f_re[np.newaxis, np.newaxis]), dim=1)))
        T_f_im = torch.cat((torch.cat((T_00_f_im, T_01_f_im[:, np.newaxis]), dim=1),
                            torch.cat((T_10_f_im[np.newaxis], T_11_f_im[np.newaxis, np.newaxis]), dim=1)))

        T = cf.Irfft_slow()(T_f_re, T_f_im) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_00, T_01, T_10, T_11 = T[:batch_size, :rank], T[:batch_size, -1], T[-1, :rank], T[-1, -1]

        # polynomial additions
        T_00 = torch.cat((T_00[:, :, :, :n2], T_00[:, :, :, n2:] + S_00[:, :, ::2] + S_00[:, :, 1::2]), dim=-1)
        T_01 = torch.cat((T_01[:, :, :n2], T_01[:, :, n2:] + S_01[:, ::2]), dim=-1)
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    # Negative step isn't supported by Pytorch
    # (https://github.com/pytorch/pytorch/issues/229) so we have to construct
    # the index explicitly.
    # reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    reverse_index = torch.arange(n - 1, -1, -1, out=torch.cuda.LongTensor())
    return T_00[:, :, :, reverse_index].squeeze(dim=2)


def krylov_transpose_multiply_fast(subdiag, v, u):
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
    assert n == n_, 'u and v must have the same last dimension'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    T_00 = u[:, np.newaxis, ..., np.newaxis] * v[np.newaxis, ..., np.newaxis]
    T_01 = u[..., np.newaxis]
    T_10 = v[..., np.newaxis]
    # T_11 = Variable(torch.ones((n, 1))).cuda()
    T_11 = Variable(torch.cuda.FloatTensor(n, 1).fill_(1.0))
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_00, S_01, S_10, S_11 = T_00, T_01, T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S1_01 = torch.cat((S_01[:, 1::2], torch.zeros_like(S_01[:, 1::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_01, S1_11[np.newaxis]))

        # polynomial multiplications
        S_f = cf.Rfft.apply(S)
        S0_10_f, S0_11_f, S1_01_f, S1_11_f = S_f[:rank], S_f[rank], S_f[rank+1:rank+1+batch_size], S_f[-1]
        # T_00_f = cf.ComplexMult.apply(S1_01_f[:, np.newaxis], S0_10_f[np.newaxis])
        # T_01_f = cf.ComplexMult.apply(S1_01_f, S0_11_f)
        # T_10_f = cf.ComplexMult.apply(S1_11_f, S0_10_f)
        # T_11_f = cf.ComplexMult.apply(S1_11_f, S0_11_f)
        T_00_f = cf.complex_mult_(S1_01_f[:, np.newaxis], S0_10_f[np.newaxis])
        T_01_f = cf.complex_mult_(S1_01_f, S0_11_f)
        T_10_f = cf.complex_mult_(S1_11_f, S0_10_f)
        T_11_f = cf.complex_mult_(S1_11_f, S0_11_f)

        T_f = torch.cat((torch.cat((T_00_f, T_01_f[:, np.newaxis]), dim=1),
                         torch.cat((T_10_f[np.newaxis], T_11_f[np.newaxis, np.newaxis]), dim=1)))

        T = cf.Irfft.apply(T_f) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_00, T_01, T_10, T_11 = T[:batch_size, :rank], T[:batch_size, -1], T[-1, :rank], T[-1, -1]

        # polynomial additions
        T_00 = torch.cat((T_00[:, :, :, :n2], T_00[:, :, :, n2:] + S_00[:, :, ::2] + S_00[:, :, 1::2]), dim=-1)
        T_01 = torch.cat((T_01[:, :, :n2], T_01[:, :, n2:] + S_01[:, ::2]), dim=-1)
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    # Negative step isn't supported by Pytorch
    # (https://github.com/pytorch/pytorch/issues/229) so we have to construct
    # the index explicitly.
    # reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    reverse_index = torch.arange(n - 1, -1, -1, out=torch.cuda.LongTensor())
    return T_00[:, :, :, reverse_index].squeeze(dim=2)


def krylov_multiply_by_autodiff(subdiag, v, w):
    """Multiply \sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal, using Pytorch's autodiff.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    # u = Variable(torch.zeros((batch_size, n)).cuda(), requires_grad=True)
    u = Variable(torch.cuda.FloatTensor(batch_size, n).fill_(0.0), requires_grad=True)
    prod = krylov_transpose_multiply_fast(subdiag, v, u)
    result, = torch.autograd.grad(prod, u, grad_outputs=w, retain_graph=True)
    return result

def krylov_multiply_forward(subdiag, v):
    rank, n = v.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = [None] * m
    T_10 = v[..., np.newaxis]
    # T_11 = Variable(torch.ones((n, 1))).cuda()
    T_11 = Variable(torch.cuda.FloatTensor(n, 1).fill_(1.0))
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_11[np.newaxis]))

        # polynomial multiplications
        S_f_re, S_f_im = cf.Rfft_slow()(S)
        S0_10_f_re, S0_11_f_re, S1_11_f_re = S_f_re[:rank], S_f_re[-2], S_f_re[-1]
        S0_10_f_im, S0_11_f_im, S1_11_f_im = S_f_im[:rank], S_f_im[-2], S_f_im[-1]
        S0_10_f = (S0_10_f_re, S0_10_f_im)
        S0_11_f = (S0_11_f_re, S0_11_f_im)
        S1_11_f = (S1_11_f_re, S1_11_f_im)
        save_for_backward[d] = (S0_10_f, S0_11_f)

        T_10_f_re, T_10_f_im = cf.complex_mult(S1_11_f, S0_10_f)
        T_11_f_re, T_11_f_im = cf.complex_mult(S1_11_f, S0_11_f)

        T_f_re = torch.cat((T_10_f_re, T_11_f_re[np.newaxis]))
        T_f_im = torch.cat((T_10_f_im, T_11_f_im[np.newaxis]))

        T = cf.Irfft_slow()(T_f_re, T_f_im) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_10, T_11 = T[:rank], T[-1]

        # polynomial additions
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    return save_for_backward

def krylov_multiply_forward_fast(subdiag, v):
    rank, n = v.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = [None] * m
    T_10 = v[..., np.newaxis]
    # T_11 = Variable(torch.ones((n, 1))).cuda()
    T_11 = Variable(torch.cuda.FloatTensor(n, 1).fill_(1.0))
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_11[np.newaxis]))

        # polynomial multiplications
        S_f = cf.Rfft.apply(S)
        S0_10_f, S0_11_f, S1_11_f = S_f[:rank], S_f[-2], S_f[-1]
        save_for_backward[d] = (S0_10_f, S0_11_f)

        # T_10_f = cf.ComplexMult.apply(S1_11_f, S0_10_f)
        # T_11_f = cf.ComplexMult.apply(S1_11_f, S0_11_f)
        T_10_f = cf.complex_mult_(S1_11_f, S0_10_f)
        T_11_f = cf.complex_mult_(S1_11_f, S0_11_f)

        T_f = torch.cat((T_10_f, T_11_f[np.newaxis]))

        T = cf.Irfft.apply(T_f) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_10, T_11 = T[:rank], T[-1]

        # polynomial additions
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    return save_for_backward

def krylov_multiply(subdiag, v, w):
    """Multiply \sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = krylov_multiply_forward(subdiag, v)
    # reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    reverse_index = torch.arange(n - 1, -1, -1, out=torch.cuda.LongTensor())
    w = w.view(batch_size, rank, 1, n)
    # dT_00, dT_01 = w[:, :, :, reverse_index], Variable(torch.zeros((batch_size, 1, n)).cuda())
    dT_00, dT_01 = w[:, :, :, reverse_index], Variable(torch.cuda.FloatTensor(batch_size, 1, n).fill_(0.0))

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        dS_00 = Variable(torch.cuda.FloatTensor(batch_size, rank, 2 * n1, n2))
        dS_00[:, :, ::2] = dT_00[:, :, :, n2:]
        dS_00[:, :, 1::2] = dT_00[:, :, :, n2:]
        dS_01 = Variable(torch.cuda.FloatTensor(batch_size, 2 * n1, n2))
        dS_01[:, ::2] = dT_01[:, :, n2:]

        dT = torch.cat((dT_00, dT_01[:, np.newaxis]), dim=1)
        dT = dT * subdiag[(n2 - 1)::(2 * n2), np.newaxis]

        dT_f_re, dT_f_im = cf.Ihfft_slow(dT)
        dT_00_f_re, dT_01_f_re = dT_f_re[:, :rank], dT_f_re[:, -1]
        dT_00_f_im, dT_01_f_im = dT_f_im[:, :rank], dT_f_im[:, -1]
        dT_00_f = (dT_00_f_re, dT_00_f_im)
        dT_01_f = (dT_01_f_re, dT_01_f_im)

        S0_10_f, S0_11_f = save_for_backward[d]
        S0_10_f_re, S0_10_f_im = S0_10_f
        dS1_01_f_re, dS1_01_f_im = cf.complex_mult((S0_10_f_re[np.newaxis], S0_10_f_im[np.newaxis]),
                                                 dT_00_f)
        prod_re, prod_im = cf.complex_mult(S0_11_f, dT_01_f)
        dS1_01_f_re = dS1_01_f_re.sum(dim=1) + prod_re
        dS1_01_f_im = dS1_01_f_im.sum(dim=1) + prod_im

        dS1_01 = cf.Hfft_slow(dS1_01_f_re, dS1_01_f_im)
        dS_01[:, 1::2] = dS1_01[:, :, :n2]

        dT_00, dT_01 = dS_00, dS_01

    du = ((dT_00 * v[np.newaxis, :, :, np.newaxis]).sum(dim=1) + dT_01).squeeze(dim=-1)
    return du


def krylov_multiply_fast(subdiag, v, w):
    """Multiply \sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = krylov_multiply_forward_fast(subdiag, v)
    # reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    reverse_index = torch.arange(n - 1, -1, -1, out=torch.cuda.LongTensor())
    w = w.view(batch_size, rank, 1, n)
    # dT_00, dT_01 = w[:, :, :, reverse_index], Variable(torch.zeros((batch_size, 1, n)).cuda())
    dT_00, dT_01 = w[:, :, :, reverse_index], Variable(torch.cuda.FloatTensor(batch_size, 1, n).fill_(0.0))

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        dS_00 = Variable(torch.cuda.FloatTensor(batch_size, rank, 2 * n1, n2))
        dS_00[:, :, ::2] = dT_00[:, :, :, n2:]
        dS_00[:, :, 1::2] = dT_00[:, :, :, n2:]
        dS_01 = Variable(torch.cuda.FloatTensor(batch_size, 2 * n1, n2))
        dS_01[:, ::2] = dT_01[:, :, n2:]

        dT = torch.cat((dT_00, dT_01[:, np.newaxis]), dim=1)
        dT = dT * subdiag[(n2 - 1)::(2 * n2), np.newaxis]

        # dT_f = cf.Ihfft.apply(dT)
        dT_f = cf.Ihfft(dT)
        dT_00_f, dT_01_f = dT_f[:, :rank], dT_f[:, -1]

        S0_10_f, S0_11_f = save_for_backward[d]
        # dS1_01_f = cf.ComplexMult.apply(S0_10_f[np.newaxis], dT_00_f).sum(dim=1) + cf.ComplexMult.apply(S0_11_f, dT_01_f)
        dS1_01_f = cf.complex_mult_(S0_10_f[np.newaxis], dT_00_f).sum(dim=1) + cf.complex_mult_(S0_11_f, dT_01_f)

        # dS1_01 = cf.Hfft.apply(dS1_01_f)
        dS1_01 = cf.Hfft(dS1_01_f)
        dS_01[:, 1::2] = dS1_01[:, :, :n2]

        dT_00, dT_01 = dS_00, dS_01

    du = ((dT_00 * v[np.newaxis, :, :, np.newaxis]).sum(dim=1) + dT_01).squeeze(dim=-1)
    return du

def subd_mult(subd_A, subd_B, G, H, x):
    rank, n = G.shape
    batch_size = x.shape[0]
    KT_out = krylov_transpose_multiply_fast(subd_B, H, x)
    K_out = krylov_multiply_fast(subd_A, G, KT_out)
    return K_out



def test_transpose_multiply():
    m = 12
    n = 1<<m
    batch_size = 512
    rank = 3
    subdiag = Variable(torch.rand(n-1), requires_grad=True).cuda()
    A = np.diag(subdiag.data.cpu().numpy(), -1)
    u = Variable(torch.rand((batch_size, n)), requires_grad=True).cuda()
    v = Variable(torch.rand((rank, n)), requires_grad=True).cuda()
    result = krylov_transpose_multiply_fast(subdiag, v, u)
    grad,  = torch.autograd.grad(torch.sum(result), v, retain_graph=True)
    grad = grad.data.cpu().numpy()
    result = result.data.cpu().numpy()
    result1 = krylov_transpose_multiply(subdiag, v, u)
    grad1, = torch.autograd.grad(torch.sum(result1), v, retain_graph=True)
    grad1 = grad1.data.cpu().numpy()
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
    print(np.max(abs(result - result1)))
    print(np.mean(abs(result - result1)))
    print(np.max(abs(grad - grad1)))
    print(np.mean(abs(grad - grad1)))
    print(np.max(abs(result1 - result2)))
    print(np.mean(abs(result1 - result2)))
    print(np.max(abs(result3 - result2)))
    print(np.mean(abs(result3 - result2)))
    print(np.max(abs(result4 - result2)))
    print(np.mean(abs(result4 - result2)))

def test_multiply():
    m = 12
    n = 1 << m
    batch_size = 512
    rank = 3
    subdiag = Variable(torch.rand(n-1), requires_grad=True).cuda()
    A = np.diag(subdiag.data.cpu().numpy(), -1)
    u = Variable(torch.rand((batch_size, n)), requires_grad=True).cuda()
    v = Variable(torch.rand((rank, n)), requires_grad=True).cuda()
    w = Variable(torch.rand((batch_size, rank, n)), requires_grad=True).cuda()
    result_fast = krylov_multiply_fast(subdiag, v, w)
    grad_fast,  = torch.autograd.grad(torch.sum(result_fast), v, retain_graph=True)
    grad_fast = grad_fast.data.cpu().numpy()
    result_fast = result_fast.data.cpu().numpy()
    result = krylov_multiply(subdiag, v, w)
    grad, = torch.autograd.grad(torch.sum(result), v, retain_graph=True)
    grad = grad.data.cpu().numpy()
    result = result.data.cpu().numpy()
    # Using autodiff
    result1 = krylov_multiply_by_autodiff(subdiag, v, w)
    result1 = result1.data.cpu().numpy()
    # CPU dense multiply
    Ks = [krylov_construct(A, v.data.cpu().numpy()[i], n) for i in range(rank)]
    w_cpu = w.data.cpu().numpy()
    result2 = np.stack([w_cpu[:, i] @ Ks[i] for i in range(rank)]).sum(axis=0)
    result2 = result2.squeeze()
    np.allclose(result_fast, result)
    np.allclose(result, result1)
    np.allclose(result1, result2)
    print(np.max(abs(result_fast - result)))
    print(np.mean(abs(result_fast - result)))
    print(np.max(abs(grad_fast - grad)))
    print(np.mean(abs(grad_fast - grad)))
    print(np.max(abs(result - result1)))
    print(np.mean(abs(result - result1)))
    print(np.max(abs(result1 - result2)))
    print(np.mean(abs(result1 - result2)))

    # Combine transpose multiply follow by non-transpose multiply
    result = krylov_multiply_fast(subdiag, v, krylov_transpose_multiply_fast(subdiag, v, u))

def test_misc():
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

    u = Variable(torch.rand((1, 8)), requires_grad=True).cuda()
    re, im = cf.Rfft_slow(u)
    t = cf.Irfft_slow(re, im)
    t1_re, t1_im = cf.Rfft_slow(t)

    grad, = torch.autograd.grad(torch.sum(re), u, retain_graph=True)
    w = Variable(torch.rand((1, 5)), requires_grad=True).cuda()
    ggrad = torch.autograd.grad(torch.sum(grad), u)
    torch.autograd.gradcheck(cf.Rfft_slow, (u, ))

    analytic_grad = fft.irfft(torch.ones_like(re).data, -torch.ones_like(im).data, normalize=False)

    epsilon = 1e-5
    for i in range(2):
        one_hot = Variable(torch.zeros_like(u.data))
        one_hot[0, i] = epsilon
        u_new = u + one_hot
        u_new_minus = u - one_hot
        print((torch.sum(cf.Rfft_slow(u_new)[0]) - torch.sum(cf.Rfft_slow(u_new_minus)[0])) / (2 * epsilon))

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


if __name__ == "__main__":
    test_transpose_multiply()
    test_multiply()
