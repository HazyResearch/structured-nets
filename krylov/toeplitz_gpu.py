'''Functions to multiply by a Toeplitz-like matrix.
'''
import numpy as np
import torch

from complex_utils import complex_mult, conjugate
from krylov_multiply import Krylov


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### Fast multiplication for the Toeplitz-like case

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
    assert n == n_, 'u and v must have the same last dimension'
    if f != 0.0:  # cycle version
        # Computing the roots of f
        mod = abs(f) ** (torch.arange(n, dtype=u.dtype, device=u.device) / n)
        if f > 0:
            arg = torch.stack((torch.ones(n, dtype=u.dtype, device=u.device),
                               torch.zeros(n, dtype=u.dtype, device=u.device)), dim=-1)
        else:  # Find primitive roots of -1
            angles = torch.arange(n, dtype=u.dtype, device=u.device) / n * np.pi
            arg = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        eta = mod[:, np.newaxis] * arg
        eta_inverse = (1.0 / mod)[:, np.newaxis] * conjugate(arg)
        u_f = torch.ifft(eta_inverse * u[..., np.newaxis], 1)
        v_f = torch.fft(eta * v[..., np.newaxis], 1)
        uv_f = complex_mult(u_f[:, np.newaxis], v_f[np.newaxis])
        uv = torch.fft(uv_f, 1)
        # We only need the real part of complex_mult(eta, uv)
        return eta[..., 0] * uv[..., 0] - eta[..., 1] * uv[..., 1]
    else:
        reverse_index = torch.arange(n - 1, -1, -1, dtype=torch.long, device=u.device)
        u_f = torch.rfft(torch.cat((u[..., reverse_index], torch.zeros_like(u)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        uv_f = complex_mult(u_f[:, np.newaxis], v_f[np.newaxis])
        return torch.irfft(uv_f, 1, signal_sizes=(2 * n, ))[..., reverse_index]


def toeplitz_krylov_multiply_by_autodiff(v, w, f=0.0):
    """Multiply \sum_i Krylov(Z_f, v_i) @ w_i, using Pytorch's autodiff.
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
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'

    u = torch.zeros((batch_size, n), dtype=v.dtype, device=v.device, requires_grad=True)
    prod = toeplitz_krylov_transpose_multiply(v, u, f)
    result, = torch.autograd.grad(prod, u, grad_outputs=w, create_graph=True)
    return result


def toeplitz_krylov_multiply(v, w, f=0.0):
    """Multiply \sum_i Krylov(Z_f, v_i) @ w_i.
    Parameters:
        v: (rank, n)
        w: (batch_size, rank, n)
        f: real number
    Returns:
        product: (batch, n)
    """
    _, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    if f != 0.0:  # cycle version
        # Computing the roots of f
        mod = abs(f) ** (torch.arange(n, dtype=w.dtype, device=w.device) / n)
        if f > 0:
            arg = torch.stack((torch.ones(n, dtype=w.dtype, device=w.device),
                               torch.zeros(n, dtype=w.dtype, device=w.device)), dim=-1)
        else:  # Find primitive roots of -1
            angles = torch.arange(n, dtype=w.dtype, device=w.device) / n * np.pi
            arg = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        eta = mod[:, np.newaxis] * arg
        eta_inverse = (1.0 / mod)[:, np.newaxis] * conjugate(arg)
        w_f = torch.fft(eta * w[..., np.newaxis], 1)
        v_f = torch.fft(eta * v[..., np.newaxis], 1)
        wv_sum_f = complex_mult(w_f, v_f).sum(dim=1)
        wv_sum = torch.ifft(wv_sum_f, 1)
        # We only need the real part of complex_mult(eta_inverse, wv_sum)
        return eta_inverse[..., 0] * wv_sum[..., 0] - eta_inverse[..., 1] - wv_sum[..., 1]
    else:
        w_f = torch.rfft(torch.cat((w, torch.zeros_like(w)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        wv_sum_f = complex_mult(w_f, v_f).sum(dim=1)
        return torch.irfft(wv_sum_f, 1, signal_sizes=(2 * n, ))[..., :n]


def toeplitz_mult(G, H, x, cycle=True):
    """Multiply \sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Parameters:
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    # f = (1,-1) if cycle else (1,1)
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
    rank, n  = v.shape
    a = torch.arange(n, dtype=torch.long, device=v.device)
    b = -a
    indices = a[:, np.newaxis] + b[np.newaxis]
    # Pytorch's advanced indexing (as of 0.4.0) is wrong for negative indices when combined with basic indexing.
    # https://github.com/pytorch/pytorch/issues/7156
    # So we have to make the indices positive.
    K = v[:, indices % n]
    K[:, indices < 0] *= f
    return K


def toeplitz_mult_slow(G, H, x, cycle=True):
    """Multiply \sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
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
    assert G.shape == H.shape, 'G and H must have the same shape'
    rank, n = G.shape
    f = (1, -1) if cycle else (0, 0)
    krylovs = [(Krylov(toeplitz_Z_f_linear_map(f[0]), G[i]), Krylov(toeplitz_Z_f_linear_map(f[1]), H[i]).t()) for i in range(rank)]
    prods = [K[0] @ (K[1] @ x.t()) for K in krylovs]
    return sum(prods).t()


def toeplitz_mult_slow_fast(G, H, x, cycle=True):
    """Multiply \sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
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


def test_toeplitz_mult():
    v = torch.tensor([[0,1,0,-1],[0,1,2,3]], dtype=torch.float, device=device, requires_grad=True)
    u = torch.tensor([[1,1,1,1],[0,1,2,3]], dtype=torch.float, device=device, requires_grad=True)

    w = toeplitz_krylov_transpose_multiply(v, u, f=-1)
    # output:
    # [[[ 0 2  2 0]
    #   [ 6 0 -4 -6]]

    #  [[ -2 2 4  2]
    #   [ 14 8 0 -8]]]

    toeplitz_mult(v, v, u)
    toeplitz_mult_slow(v, v, u)
    # output:
    # array([[-16., -20.,  -4.,  16.],
    #        [ 16.,  -8.,  12.,  64.]])

    toeplitz_mult(v, v, u, cycle=False)
    toeplitz_mult_slow(v, v, u, cycle=False)
    # output:
    # array([[ 0.,  6., 16., 26.],
    #        [ 0., 12., 38., 66.]])

    m = 10
    n = 1<<m
    batch_size = 50
    rank = 16
    u = torch.rand((batch_size, n), requires_grad=True, device=device)
    v = torch.rand((rank, n), requires_grad=True, device=device)
    result = toeplitz_mult(v, v, u, cycle=True)
    grad, = torch.autograd.grad(result.sum(), v, retain_graph=True)
    result_slow = toeplitz_mult_slow(v, v, u, cycle=True)
    grad_slow, = torch.autograd.grad(result_slow.sum(), v, retain_graph=True)
    result_slow_fast = toeplitz_mult_slow_fast(v, v, u, cycle=True)
    grad_slow_fast, = torch.autograd.grad(result_slow_fast.sum(), v, retain_graph=True)
    # These max and mean errors should be small
    print((result - result_slow).abs().max().item())
    print((result - result_slow).abs().mean().item())
    print((grad - grad_slow).abs().max().item())
    print((grad - grad_slow).abs().mean().item())
    print((result - result_slow_fast).abs().max().item())
    print((result - result_slow_fast).abs().mean().item())
    print((grad - grad_slow_fast).abs().max().item())
    print((grad - grad_slow_fast).abs().mean().item())


def test_memory():
    """Memory stress test to make sure there's no memory leak.
    """
    for _ in range(10000):
        a = torch.empty((2,4096), dtype=torch.float, device=device, requires_grad=True)
        b = torch.empty((2,4096), dtype=torch.float, device=device, requires_grad=True)
        c = toeplitz_mult(a, a, b)
        g, = torch.autograd.grad(torch.sum(c), a, retain_graph=True)


if __name__ == '__main__':
    test_toeplitz_mult()
    # test_memory()
