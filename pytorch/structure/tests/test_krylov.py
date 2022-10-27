import torch
from mle.structure.krylov import (
    CycleDownMultCuda,
    Krylov,
    krylov_multiply,
    krylov_multiply_by_autodiff,
    krylov_subdiag_fast,
    krylov_transpose_multiply,
    krylov_transpose_multiply_conv,
    subdiag_linear_map,
    subdiag_mult,
    subdiag_mult_conv,
    subdiag_mult_cuda,
    subdiag_mult_slow,
    subdiag_mult_slow_fast,
    subdiag_mult_slow_old,
    tridiag_linear_map,
    tridiag_linear_map_slow,
)
from torch.nn import functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def test_cycle_down_mult():
    if not torch.cuda.is_available():
        return

    n = 1 << 10
    rank = 16
    subdiag = torch.rand(n, requires_grad=True, device=device)
    v = torch.rand((rank, n), requires_grad=True, device=device)
    z = CycleDownMultCuda.apply(subdiag, v)
    y = torch.cat((subdiag[0] * v[..., -1:], subdiag[1:] * v[..., :-1]), dim=-1)

    torch.testing.assert_close(z, y)

    grad_output = torch.rand_like(y)
    gs, gv = torch.autograd.grad(y, (subdiag, v), grad_output, retain_graph=True)
    zs, zv = torch.autograd.grad(z, (subdiag, v), grad_output, retain_graph=True)

    torch.testing.assert_close(zs, gs)
    torch.testing.assert_close(zv, gv)


def test_krylov_transpose_multiply():
    m = 10
    n = 1 << m
    batch_size = 50
    rank = 16

    subdiag = torch.rand(n - 1, requires_grad=True, device=device)
    u = torch.rand((batch_size, n), requires_grad=True, device=device)
    v = torch.rand((rank, n), requires_grad=True, device=device)

    # Fast algorithm on GPU
    result = krylov_transpose_multiply(subdiag, v, u)

    # Version based on pytorch convolution
    result_conv = krylov_transpose_multiply_conv(subdiag, v, u)
    # torch.testing.assert_close(result, result_conv)
    result = result_conv

    # Compare with old algorithm
    # result_old = krylov_transpose_multiply_old(subdiag, v, u)
    # torch.testing.assert_close(result, result_old)

    # CPU dense multiply
    A = torch.diag(subdiag, diagonal=-1)
    Ks = [krylov_construct(A, v[i], n) for i in range(rank)]
    result_cpu = torch.stack([u @ K.T for K in Ks])
    result_cpu = result_cpu.swapaxes(0, 1).squeeze()
    result_cpu = torch.tensor(result_cpu, dtype=torch.float, device=device)
    torch.testing.assert_close(result, result_cpu)

    # GPU dense multiply
    Ks_gpu_dense = [torch.tensor(K, dtype=torch.float, device=device) for K in Ks]
    result_gpu_dense = torch.stack([u @ K.t() for K in Ks_gpu_dense])
    result_gpu_dense = result_gpu_dense.transpose(0, 1).squeeze()
    torch.testing.assert_close(result, result_gpu_dense)

    # Explicit construction on GPU
    Ks_gpu = Krylov(subdiag_linear_map(subdiag), v)
    result_gpu = (u @ Ks_gpu).transpose(0, 1)
    torch.testing.assert_close(result, result_gpu)

    # Explicit construction on GPU, but faster
    Ks_gpu_fast = krylov_subdiag_fast(subdiag, v)
    result_gpu_fast = (u @ Ks_gpu_fast).transpose(0, 1)
    torch.testing.assert_close(result, result_gpu_fast)

    # Compute grads as well
    (grad,) = torch.autograd.grad(result.sum(), subdiag, retain_graph=True)
    (grad_gpu,) = torch.autograd.grad(result_gpu.sum(), subdiag, retain_graph=True)
    (grad_gpu_fast,) = torch.autograd.grad(
        result_gpu_fast.sum(), subdiag, retain_graph=True
    )

    torch.testing.assert_close(grad, grad_gpu)
    torch.testing.assert_close(grad, grad_gpu_fast)


def test_krylov_multiply():
    m = 10
    n = 1 << m
    batch_size = 50
    rank = 16
    subdiag = torch.rand(n - 1, requires_grad=True, device=device)
    v = torch.rand((rank, n), requires_grad=True, device=device)
    w = torch.rand((batch_size, rank, n), requires_grad=True, device=device)

    # Fast algorithm on GPU
    result = krylov_multiply(subdiag, v, w)
    (grad,) = torch.autograd.grad(result.sum(), subdiag, retain_graph=True)

    # Using autodiff
    result_autodiff = krylov_multiply_by_autodiff(subdiag, v, w)
    torch.testing.assert_close(result, result_autodiff)

    (grad_autodiff,) = torch.autograd.grad(
        result_autodiff.sum(), subdiag, retain_graph=True
    )
    torch.testing.assert_close(grad, grad_autodiff)

    # CPU dense multiply
    A = torch.diag(subdiag, diagonal=-1)
    Ks = [krylov_construct(A, v[i], n) for i in range(rank)]
    result_cpu = (
        torch.stack([w[:, i] @ Ks[i] for i in range(rank)]).sum(axis=0).squeeze()
    )
    result_cpu = torch.tensor(result_cpu, dtype=torch.float, device=device)
    torch.testing.assert_close(result, result_cpu)

    # Explicit construction on GPU
    Ks_gpu = Krylov(subdiag_linear_map(subdiag), v)
    result_gpu = (w.transpose(0, 1) @ Ks_gpu.transpose(1, 2)).sum(dim=0)
    torch.testing.assert_close(result, result_gpu)
    (grad_gpu,) = torch.autograd.grad(result_gpu.sum(), subdiag, retain_graph=True)
    torch.testing.assert_close(grad, grad_gpu)

    # Explicit construction on GPU, but faster
    Ks_gpu_fast = krylov_subdiag_fast(subdiag, v)
    result_gpu_fast = (w.transpose(0, 1) @ Ks_gpu_fast.transpose(1, 2)).sum(dim=0)
    torch.testing.assert_close(result, result_gpu_fast)
    (grad_gpu_fast,) = torch.autograd.grad(
        result_gpu_fast.sum(), subdiag, retain_graph=True
    )
    torch.testing.assert_close(grad, grad_gpu_fast)


def test_subdiag_mult():
    m = 8
    n = 1 << m
    batch_size = 50
    rank = 16
    subdiag = torch.rand(n - 1, requires_grad=True, device=device)
    u = torch.rand((batch_size, n), requires_grad=True, device=device)
    v = torch.rand((rank, n), requires_grad=True, device=device)

    K = Krylov(subdiag_linear_map(subdiag, 1.0), v)
    K_fast = krylov_subdiag_fast(subdiag, v, upper_right_corner=1.0)
    torch.testing.assert_close(K, K_fast)

    result = subdiag_mult(subdiag, subdiag, v, v, u)

    result_conv = subdiag_mult_conv(subdiag, subdiag, v, v, u)
    torch.testing.assert_close(result, result_conv)

    result_slow_old = subdiag_mult_slow_old(subdiag, subdiag, v, v, u)
    torch.testing.assert_close(result, result_slow_old)

    result_slow = subdiag_mult_slow(subdiag, subdiag, v, v, u)
    torch.testing.assert_close(result, result_slow)

    result_slow_fast = subdiag_mult_slow_fast(subdiag, subdiag, v, v, u)
    torch.testing.assert_close(result, result_slow_fast)

    if torch.cuda.is_available():
        result_cuda = subdiag_mult_cuda(subdiag, subdiag, v, v, u)
        torch.testing.assert_close(result, result_cuda)

    # Test different ways to compute grad
    (grad,) = torch.autograd.grad(result.sum(), subdiag, retain_graph=True)

    (grad_slow_old,) = torch.autograd.grad(
        result_slow_old.sum(), subdiag, retain_graph=True
    )
    torch.testing.assert_close(grad, grad_slow_old)

    (grad_slow,) = torch.autograd.grad(result_slow.sum(), subdiag, retain_graph=True)
    torch.testing.assert_close(grad, grad_slow)

    (grad_slow_fast,) = torch.autograd.grad(
        result_slow_fast.sum(), subdiag, retain_graph=True
    )
    torch.testing.assert_close(grad, grad_slow_fast)

    if torch.cuda.is_available():
        (grad_cuda,) = torch.autograd.grad(
            result_cuda.sum(), subdiag, retain_graph=True
        )
        torch.testing.assert_close(grad, grad_cuda)


def test_tridiag_mult():
    m = 10
    n = 1 << m
    rank = 16
    subdiag = torch.rand(n - 1, requires_grad=True, device=device) / 2
    diag = torch.rand(n, requires_grad=True, device=device) / 2
    superdiag = torch.rand(n - 1, requires_grad=True, device=device) / 2
    v = torch.rand((rank, n), requires_grad=True, device=device)
    K = Krylov(tridiag_linear_map(subdiag, diag, superdiag, 0.5, 0.5), v)
    K_old = Krylov(tridiag_linear_map_slow(subdiag, diag, superdiag, 0.5, 0.5), v)

    torch.testing.assert_close(K, K_old)


def poly_mult_sum_benchmark(p, q):
    """Multiply and sum two sets of polynomials.
    Parameters:
        p: (batch_size, n1, n2)
        q: (rank, n1, n2)
    Output:
        o: (batch_size, rank, 2 * n2 - 1)
    """
    print(p.shape[2])

    import time

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y = F.conv1d(p, q.flip(q.dim() - 1), padding=p.shape[-1] - 1)
        _ = torch.autograd.grad(y.sum(), (p, q), retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Elapsed time conv1d: {end - start}s.")

    batch_size, rank = p.shape[0], q.shape[0]
    _, n2 = p.shape[1], p.shape[2]
    start = time.perf_counter()
    for _ in range(100):
        S = torch.cat(
            (
                torch.cat((q, p)),
                torch.zeros(
                    (rank + batch_size, p.shape[1], p.shape[2]),
                    dtype=q.dtype,
                    device=q.device,
                ),
            ),
            dim=-1,
        )
        S_f = torch.rfft(S, 1)
        S0_10_f, S1_01_f = S_f[:rank], S_f[rank : rank + batch_size]
        prod = (S1_01_f[:, None, ..., None] * S0_10_f[None, ..., None, :]).sum(dim=2)
        T_00_f_sum = torch.stack(
            (prod[..., 0, 0] - prod[..., 1, 1], prod[..., 0, 1] + prod[..., 1, 0]),
            dim=-1,
        )
        T_00_sum = torch.irfft(T_00_f_sum, 1, signal_sizes=(2 * n2,))[..., :-1]
        _ = torch.autograd.grad(T_00_sum.sum(), (p, q), retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Elapsed time FFT: {end - start}s.\n")

    return F.conv1d(p, q.flip(q.dim() - 1), padding=p.shape[-1] - 1)


def poly_mult_sum_backward_benchmark(grad, q):
    """Backward pass of multiplying and summing two sets of polynomials.
    Parameters:
        grad: (batch_size, rank, 2 * n2 - 1)
        q: (rank, n1, n2)
    Output:
        dp: (batch_size, n1, n2)
    """
    print(q.shape[2])

    import time

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        dp = F.conv_transpose1d(grad, q.flip(2), padding=q.shape[-1] - 1)
        _ = torch.autograd.grad(dp.sum(), (grad, q), retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Elapsed time conv1d: {end - start}s.")

    batch_size, rank = grad.shape[0], q.shape[0]
    _, n2 = q.shape[1], q.shape[2]
    start = time.perf_counter()
    for _ in range(100):
        dT_00_sum = torch.cat(
            (
                grad,
                torch.zeros(
                    (batch_size, rank, 1), dtype=grad.dtype, device=grad.device
                ),
            ),
            dim=-1,
        )
        dT_00_sum_f = torch.rfft(dT_00_sum, 1)
        S0_10_f = torch.rfft(torch.cat((q, torch.zeros_like(q)), dim=-1), 1)
        # dS1_01_f = complex_mult(conjugate(S0_10_f), dT_00_sum_f[:, :, np.newaxis]).sum(dim=1)
        # Manually doing complex multiply
        prod = (S0_10_f[..., None] * dT_00_sum_f[:, :, None, :, None, :]).sum(dim=1)
        dS1_01_f = torch.stack(
            (prod[..., 0, 0] + prod[..., 1, 1], prod[..., 0, 1] - prod[..., 1, 0]),
            dim=-1,
        )
        dp = torch.irfft(dS1_01_f, 1, signal_sizes=(2 * n2,))[:, :, :n2]
        _ = torch.autograd.grad(dp.sum(), (grad, q), retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Elapsed time FFT: {end - start}s.\n")

    return F.conv_transpose1d(grad, q.flip(2), padding=q.shape[-1] - 1)


def krylov_construct(A, v, m):
    # Note: This version is different from the krylov_construct in toeplitz_cpu
    n = v.shape[0]
    assert A.shape == (n, n)
    subd = torch.diagonal(A, -1)

    K = torch.zeros(size=(m, n), device=A.device)
    K[0, :] = v
    for i in range(1, m):
        K[i, 1:] = subd * K[i - 1, :-1]
    return K
