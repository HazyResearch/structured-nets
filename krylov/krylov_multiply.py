import functools
import numpy as np

import torch

from triextrafat import krylov_construct
# from triXXF import KrylovTransposeMultiply

from complex_utils import complex_mult, conjugate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    T_11 = torch.ones((n, 1), device=T_00.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_00, S_01, S_10, S_11 = T_00, T_01, T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S1_01 = torch.cat((S_01[:, 1::2], torch.zeros_like(S_01[:, 1::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_01, S1_11[np.newaxis]))

        # polynomial multiplications
        S_f = torch.rfft(S, 1)
        S0_10_f, S0_11_f, S1_01_f, S1_11_f = S_f[:rank], S_f[rank], S_f[rank+1:rank+1+batch_size], S_f[-1]
        T_00_f = complex_mult(S1_01_f[:, np.newaxis], S0_10_f[np.newaxis])
        T_01_f = complex_mult(S1_01_f, S0_11_f)
        T_10_f = complex_mult(S1_11_f, S0_10_f)
        T_11_f = complex_mult(S1_11_f, S0_11_f)

        T_f = torch.cat((torch.cat((T_00_f, T_01_f[:, np.newaxis]), dim=1),
                         torch.cat((T_10_f[np.newaxis], T_11_f[np.newaxis, np.newaxis]), dim=1)))

        T = torch.irfft(T_f, 1, signal_sizes=(2 * n2, )) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_00, T_01, T_10, T_11 = T[:batch_size, :rank], T[:batch_size, -1], T[-1, :rank], T[-1, -1]

        # polynomial additions
        T_00 = torch.cat((T_00[:, :, :, :n2], T_00[:, :, :, n2:] + S_00[:, :, ::2] + S_00[:, :, 1::2]), dim=-1)
        T_01 = torch.cat((T_01[:, :, :n2], T_01[:, :, n2:] + S_01[:, ::2]), dim=-1)
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    # Negative step isn't supported by Pytorch
    # (https://github.com/pytorch/pytorch/issues/229) so we have to construct
    # the index explicitly.
    reverse_index = torch.arange(n - 1, -1, -1, dtype=torch.long, device=T_00.device)
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

    u = torch.zeros((batch_size, n), dtype=v.dtype, device=v.device, requires_grad=True)
    prod = krylov_transpose_multiply(subdiag, v, u)
    result, = torch.autograd.grad(prod, u, grad_outputs=w, create_graph=True)
    return result


def krylov_multiply_forward_(subdiag, v):
    rank, n = v.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = [None] * m
    T_10 = v[..., np.newaxis]
    T_11 = torch.ones((n, 1), device=T_10.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_11[np.newaxis]))

        # polynomial multiplications
        S_f = torch.rfft(S, 1)
        S0_10_f, S0_11_f, S1_11_f = S_f[:rank], S_f[-2], S_f[-1]
        save_for_backward[d] = (S0_10_f, S0_11_f)

        T_10_f = complex_mult(S1_11_f, S0_10_f)
        T_11_f = complex_mult(S1_11_f, S0_11_f)

        T_f = torch.cat((T_10_f, T_11_f[np.newaxis]))

        T = torch.irfft(T_f, 1, signal_sizes=(2 * n2, )) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
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

    save_for_backward = krylov_multiply_forward_(subdiag, v)
    reverse_index = torch.arange(n - 1, -1, -1, dtype=torch.long, device=w.device)
    w = w[:, :, np.newaxis, :]
    dT_00, dT_01 = w[:, :, :, reverse_index], torch.zeros((batch_size, 1, n), dtype=w.dtype, device=w.device)

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        dS_00 = torch.empty((batch_size, rank, 2 * n1, n2), device=w.device)
        dS_00[:, :, ::2] = dT_00[:, :, :, n2:]
        dS_00[:, :, 1::2] = dT_00[:, :, :, n2:]
        dS_01 = torch.empty((batch_size, 2 * n1, n2), device=w.device)
        dS_01[:, ::2] = dT_01[:, :, n2:]

        dT = torch.cat((dT_00, dT_01[:, np.newaxis]), dim=1)
        dT = dT * subdiag[(n2 - 1)::(2 * n2), np.newaxis]

        dT_f = torch.rfft(dT, 1) / (2 * n2)
        dT_00_f, dT_01_f = dT_f[:, :rank], dT_f[:, -1]

        S0_10_f, S0_11_f = save_for_backward[d]
        dS1_01_f = complex_mult(conjugate(S0_10_f)[np.newaxis], dT_00_f).sum(dim=1) + complex_mult(conjugate(S0_11_f), dT_01_f)

        dS1_01 = torch.irfft(dS1_01_f, 1, signal_sizes=(2 * n2, )) * (2 * n2)
        dS_01[:, 1::2] = dS1_01[:, :, :n2]

        dT_00, dT_01 = dS_00, dS_01

    du = ((dT_00 * v[np.newaxis, :, :, np.newaxis]).sum(dim=1) + dT_01).squeeze(dim=-1)
    return du


def subd_mult(subd_A, subd_B, G, H, x):
    rank, n = G.shape
    batch_size = x.shape[0]
    # print("x.shape", x.shape)
    # if not power of 2, round everything up
    # TODO: this can maybe be handled better. also should benchmark how much speed non-po2 FFT loses
    m = int(np.ceil(np.log2(n)))
    if n != 1 << m:
        n_ = 1 << m
        x_ = torch.cat((x, torch.zeros(batch_size,n_-n).cuda()), dim=-1)
        G_ = torch.cat((G, torch.zeros(rank,n_-n).cuda()), dim=-1)
        H_ = torch.cat((H, torch.zeros(rank,n_-n).cuda()), dim=-1)
        subd_A_ = torch.cat((subd_A, torch.zeros(n_-n).cuda()))
        subd_B_ = torch.cat((subd_B, torch.zeros(n_-n).cuda()))
        KT_out_ = krylov_transpose_multiply(subd_B_, H_, x_)
        K_out_ = krylov_multiply(subd_A_, G_, KT_out_)
        return K_out_[:, :n]

    KT_out = krylov_transpose_multiply(subd_B, H, x)
    K_out = krylov_multiply(subd_A, G, KT_out)
    return K_out

def subd_mult_slow_old(subd_A, subd_B, G, H, x):
    rank, n = G.shape
    linear_map_A = functools.partial(shift_subdiag, subd_A)
    linear_map_B = functools.partial(shift_subdiag, subd_B)
    krylovs = [(Krylov(linear_map_A, G[i]), Krylov(linear_map_B, H[i]).t()) for i in range(rank)]
    prods = [K[0] @ (K[1] @ x.t()) for K in krylovs]
    return sum(prods).t()

def subd_mult_slow(subd_A, subd_B, G, H, x, corner_A=0.0, corner_B=0.0):
    if G.shape[0] == 1:  # specialized code for rank=1, giving 2x speedup.
        K_G = Krylov(subdiag_linear_map(subd_A, corner_A), G[0])
        K_H = Krylov(subdiag_linear_map(subd_B, corner_B), H[0])
        return (x @ K_H) @ K_G.t()
    else:
        K_G = Krylov(subdiag_linear_map(subd_A, corner_A), G)
        K_H = Krylov(subdiag_linear_map(subd_B, corner_B), H)
        return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)

def subd_mult_slow_fast(subd_A, subd_B, G, H, x):
    K_G, K_H = krylov_subdiag_fast(subd_A, G), krylov_subdiag_fast(subd_B, H)
    return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)

def trid_mult_slow(subd_A, diag_A, supd_A, subd_B, diag_B, supd_B, G, H, x, corners_A=(0.0, 0.0), corners_B=(0.0, 0.0)):
    if G.shape[0] == 1:  # specialized code for rank=1, giving 2x speedup.
        K_G = Krylov(tridiag_linear_map(subd_A, diag_A, supd_A, *corners_A), G[0])
        K_H = Krylov(tridiag_linear_map(subd_B, diag_B, supd_B, *corners_B), H[0])
        return (x @ K_H) @ K_G.t()
    else:
        K_G = Krylov(tridiag_linear_map(subd_A, diag_A, supd_A, *corners_A), G)
        K_H = Krylov(tridiag_linear_map(subd_B, diag_B, supd_B, *corners_B), H)
        return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)


def test_transpose_multiply():
    m = 12
    n = 1<<m
    batch_size = 50
    rank = 16
    subdiag = torch.rand(n-1, requires_grad=True, device=device)
    A = np.diag(subdiag.data.cpu().numpy(), -1)
    u = torch.rand((batch_size, n), requires_grad=True, device=device)
    v = torch.rand((rank, n), requires_grad=True, device=device)
    result = krylov_transpose_multiply(subdiag, v, u)
    grad,  = torch.autograd.grad(torch.sum(result), v, retain_graph=True)
    grad = grad.data.cpu().numpy()
    result = result.data.cpu().numpy()
    # CPU dense multiply
    Ks = [krylov_construct(A, v.data.cpu().numpy()[i], n) for i in range(rank)]
    u_cpu = u.data.cpu().numpy()
    result2 = np.stack([u_cpu @ K.T for K in Ks])
    result2 = result2.swapaxes(0, 1).squeeze()
    # GPU dense multiply
    Ks_pytorch = [torch.Tensor(K).cuda() for K in Ks]
    result3 = torch.stack([u @ K.t() for K in Ks_pytorch])
    result3 = result3.data.cpu().numpy().swapaxes(0, 1).squeeze()
    # Explicit construction on GPU
    linear_fn = functools.partial(shift_subdiag, subdiag)
    Ks_gpu = [Krylov(linear_fn, v_) for v_ in v]
    result4 = torch.stack([u @ K for K in Ks_gpu])
    result4 = result4.data.cpu().numpy().swapaxes(0, 1).squeeze()
    # Explicit construction on GPU, but faster
    Ks_gpu = [krylov_construct_fast(subdiag, v_, f=0.0) for v_ in v]
    result5 = torch.stack([u @ K for K in Ks_gpu])
    result5 = result5.data.cpu().numpy().swapaxes(0, 1).squeeze()
    print(np.max(abs(result - result2)))
    print(np.mean(abs(result - result2)))
    print(np.max(abs(result3 - result2)))
    print(np.mean(abs(result3 - result2)))
    print(np.max(abs(result4 - result2)))
    print(np.mean(abs(result4 - result2)))
    print(np.max(abs(result5 - result)))
    print(np.mean(abs(result5 - result)))


def test_multiply():
    m = 10
    n = 1 << m
    batch_size = 50
    rank = 16
    subdiag = torch.rand(n-1, requires_grad=True, device=device)
    A = np.diag(subdiag.data.cpu().numpy(), -1)
    u = torch.rand((batch_size, n), requires_grad=True, device=device)
    v = torch.rand((rank, n), requires_grad=True, device=device)
    w = torch.rand((batch_size, rank, n), requires_grad=True, device=device)
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
    assert np.allclose(result, result1)
    assert np.allclose(result1, result2)
    print(np.max(abs(result - result1)))
    print(np.mean(abs(result - result1)))
    print(np.max(abs(result1 - result2)))
    print(np.mean(abs(result1 - result2)))

    # Combine transpose multiply follow by non-transpose multiply
    result = krylov_multiply_mine(subdiag, v, krylov_transpose_multiply_mine(subdiag, v, u))

    diag = torch.rand(n, requires_grad=True, device=device)
    superdiag = torch.rand(n-1, requires_grad=True, device=device)

    result = subd_mult(subdiag, subdiag, v, v, u)
    grad,  = torch.autograd.grad(result.sum(), subdiag, retain_graph=True)
    result_slow_old = subd_mult_slow_old(subdiag, subdiag, v, v, u)
    grad_slow_old,  = torch.autograd.grad(result_slow_old.sum(), subdiag, retain_graph=True)
    result_slow_fast = subd_mult_slow_fast(subdiag, subdiag, v, v, u)
    grad_slow_fast,  = torch.autograd.grad(result_slow_fast.sum(), subdiag, retain_graph=True)
    result_slow = subd_mult_slow(subdiag, subdiag, v, v, u)
    grad_slow,  = torch.autograd.grad(result_slow.sum(), subdiag, retain_graph=True)
    print(torch.max(torch.abs(result - result_slow_old)).item())
    print(torch.mean(torch.abs(result - result_slow_old)).item())
    print((grad - grad_slow_old).abs().max().item())
    print((grad - grad_slow_old).abs().mean().item())
    print(torch.max(torch.abs(result - result_slow)).item())
    print(torch.mean(torch.abs(result - result_slow)).item())
    print((grad - grad_slow).abs().max().item())
    print((grad - grad_slow).abs().mean().item())
    print(torch.max(torch.abs(result_slow_fast - result)).item())
    print(torch.mean(torch.abs(result_slow_fast - result)).item())
    print((grad - grad_slow_fast).abs().max().item())
    print((grad - grad_slow_fast).abs().mean().item())

    trid_slow = trid_mult_slow(subdiag, diag, superdiag, subdiag, diag, superdiag, v, v, u)

def test_misc():
    pass
    # epsilon = 1e-5
    # for i in range(2):
    #     one_hot = Variable(torch.zeros_like(u.data))
    #     one_hot[0, i] = epsilon
    #     u_new = u + one_hot
    #     u_new_minus = u - one_hot
    #     print((torch.sum(cf.Rfft_slow(u_new)[0]) - torch.sum(cf.Rfft_slow(u_new_minus)[0])) / (2 * epsilon))

def Krylov(linear_map, v, n=None):
    if n is None:
        n = v.size(-1)
    cols = [v]
    for _ in range(n - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)

def shift_subdiag(subdiag, v, f=0.0):
    return torch.cat((f * v[[-1]], subdiag * v[:-1]))

def subdiag_linear_map(subdiag, upper_right_corner=0.0):
    n = subdiag.size(0) + 1
    shift_down = torch.arange(-1, n - 1, dtype=torch.long, device=v.device) % n
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    return lambda v: subdiag_extended * v[..., shift_down]

def tridiag_linear_map(subdiag, diag, superdiag, upper_right_corner=0.0, lower_left_corner=0.0):
    n = diag.size(0)
    shift_none = torch.arange(n, dtype=torch.long, device=v.device)
    shift_down = (shift_none - 1) % n
    shift_up = (shift_none + 1) % n
    shifts = torch.stack((shift_down, shift_none, shift_up))
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    superdiag_extended = torch.cat((superdiag, torch.tensor([lower_left_corner], dtype=subdiag.dtype, device=subdiag.device)))
    diags = torch.stack((subdiag_extended, diag, superdiag_extended))
    return lambda v: (diags * v[..., shifts]).sum(dim=-2)

def tridiag_linear_map_slow(subdiag, diag, superdiag, upper_right_corner=0.0, lower_left_corner=0.0):
    return lambda v: torch.cat((upper_right_corner * v[..., -1:], subdiag * v[..., :-1]), dim=-1) + diag * v + torch.cat((superdiag * v[..., 1:], lower_left_corner * v[..., :1]), dim=-1)


def krylov_subdiag_fast(subdiag, v, f=0.0):
    rank, n  = v.shape
    a = torch.arange(n, dtype=torch.long, device=v.device)
    b = -a
    # Pytorch's advanced indexing (as of 0.4.0) is wrong for negative indices when combined with basic indexing.
    # So we have to make the indices positive.
    indices = (a[:, np.newaxis] + b[np.newaxis]) % n
    v_circulant = v[:, indices]
    subdiag_extended = torch.cat((torch.tensor([f], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    subdiag_circulant = subdiag_extended[indices]
    subdiag_cumprod = subdiag_circulant.cumprod(dim=1)
    K = v_circulant
    K[:, :, 1:] *= subdiag_cumprod[:, :-1]
    return K

def krylov_subdiag_test():
    m = 10
    n = 1 << m
    subdiag = torch.rand(n - 1, requires_grad=True, device=device)
    v = torch.rand((16, n), requires_grad=True, device=device)
    K = Krylov(subdiag_linear_map(subdiag, 1.0), v)
    K_fast = krylov_subdiag_fast(subdiag, v, f=1.0)
    print((K - K_fast).abs().max().item())

def krylov_tridiag_test():
    m = 10
    n = 1 << m
    subdiag = torch.rand(n-1, requires_grad=True, device=device) / 2
    diag = torch.rand(n, requires_grad=True, device=device) / 2
    superdiag = torch.rand(n-1, requires_grad=True, device=device) / 2
    v = torch.rand((3, n), requires_grad=True, device=device)
    K = Krylov(tridiag_linear_map(subdiag, diag, superdiag, 0.5, 0.5), v)
    K_old = Krylov(tridiag_linear_map_slow(subdiag, diag, superdiag, 0.5, 0.5), v)
    print((K - K_old).abs().max().item())

if __name__ == "__main__":
    test_transpose_multiply()
    test_multiply()
    # krylov_subdiag_test()
    # krylov_tridiag_test()
