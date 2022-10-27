import torch
from mle.structure.toeplitz import (
    toeplitz_krylov_transpose_multiply,
    toeplitz_mult_slow,
    toeplitz_mult_slow_fast,
)
from mle.structure.toeplitz_cpu import KT_Toeplitz, toeplitz_mult

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def test_toeplitz():
    v = torch.tensor([[0, 1, 0, -1], [0, 1, 2, 3]], device=device, dtype=torch.float)
    u = torch.tensor([[1, 1, 1, 1], [0, 1, 2, 3]], device=device, dtype=torch.float)

    torch.testing.assert_close(
        KT_Toeplitz(4, -1, 2, 2)(v, u),
        torch.tensor(
            [[[0, 2, 2, 0], [6, 0, -4, -6]], [[-2, 2, 4, 2], [14, 8, 0, -8]]],
            device=device,
            dtype=torch.float,
        ),
    )

    torch.testing.assert_close(
        KT_Toeplitz(4, 0, 2, 2)(v, u),
        torch.tensor(
            [[[0, 1, 1, 0], [6, 3, 1, 0]], [[-2, 2, 3, 0], [14, 8, 3, 0]]],
            device=device,
            dtype=torch.float,
        ),
    )

    w = torch.tensor(
        [[-16.0, -20.0, -4.0, 16.0], [16.0, -8.0, 12.0, 64.0]],
        device=device,
    )
    torch.testing.assert_close(
        w, toeplitz_mult_slow(v, v, u).to(device).to(torch.float32)
    )
    torch.testing.assert_close(w, toeplitz_mult_slow_fast(v, v, u))
    torch.testing.assert_close(w, toeplitz_mult(v, v, u))

    w = torch.tensor([[0.0, 6.0, 16.0, 26.0], [0.0, 12.0, 38.0, 66.0]], device=device)
    torch.testing.assert_close(
        w, toeplitz_mult_slow(v, v, u, cycle=False).to(device).to(torch.float32)
    )
    torch.testing.assert_close(w, toeplitz_mult_slow_fast(v, v, u, cycle=False))
    torch.testing.assert_close(w, toeplitz_mult(v, v, u, cycle=False))


def test_toeplitz_mult():
    v = torch.tensor(
        [[0, 1, 0, -1], [0, 1, 2, 3]],
        dtype=torch.float,
        device=device,
        requires_grad=True,
    )
    u = torch.tensor(
        [[1, 1, 1, 1], [0, 1, 2, 3]],
        dtype=torch.float,
        device=device,
        requires_grad=True,
    )

    w = torch.tensor(
        [[[0, 2, 2, 0], [6, 0, -4, -6]], [[-2, 2, 4, 2], [14, 8, 0, -8]]],
        dtype=torch.float,
        device=device,
    )
    torch.testing.assert_close(w, toeplitz_krylov_transpose_multiply(v, u, f=-1))
    torch.testing.assert_close(
        w, KT_Toeplitz(n=v.shape[1], f=-1, batch_size=u.shape[0], rank=v.shape[0])(v, u)
    )

    w = torch.tensor(
        [[-16.0, -20.0, -4.0, 16.0], [16.0, -8.0, 12.0, 64.0]], device=device
    )
    torch.testing.assert_close(w, toeplitz_mult_slow(v, v, u).to(device))
    torch.testing.assert_close(w, toeplitz_mult_slow_fast(v, v, u))
    torch.testing.assert_close(w, toeplitz_mult(v, v, u))

    w = torch.tensor([[0.0, 6.0, 16.0, 26.0], [0.0, 12.0, 38.0, 66.0]], device=device)
    torch.testing.assert_close(w, toeplitz_mult_slow(v, v, u, cycle=False).to(device))
    torch.testing.assert_close(w, toeplitz_mult_slow_fast(v, v, u, cycle=False))
    torch.testing.assert_close(w, toeplitz_mult(v, v, u, cycle=False))

    m = 10
    n = 1 << m
    batch_size = 50
    rank = 16
    u = torch.rand(
        (batch_size, n), requires_grad=True, dtype=torch.float64, device=device
    )
    v = torch.rand((rank, n), requires_grad=True, dtype=torch.float64, device=device)

    result = toeplitz_mult(v, v, u, cycle=True)
    result_slow = toeplitz_mult_slow(v, v, u, cycle=True)
    result_slow_fast = toeplitz_mult_slow_fast(v, v, u, cycle=True)
    torch.testing.assert_close(result, result_slow, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(result, result_slow_fast, atol=1e-2, rtol=1e-2)

    (grad,) = torch.autograd.grad(result.sum(), v, retain_graph=True)
    (grad_slow,) = torch.autograd.grad(result_slow.sum(), v, retain_graph=True)
    (grad_slow_fast,) = torch.autograd.grad(
        result_slow_fast.sum(), v, retain_graph=True
    )
    torch.testing.assert_close(grad, grad_slow, rtol=1e-4, atol=1e-2)
    torch.testing.assert_close(grad, grad_slow_fast, rtol=1e-4, atol=1e-2)


def _test_memory():
    """Memory stress test to make sure there's no memory leak."""
    for _ in range(10000):
        a = torch.empty((2, 4096), dtype=torch.float, device=device, requires_grad=True)
        b = torch.empty((2, 4096), dtype=torch.float, device=device, requires_grad=True)
        c = toeplitz_mult(a, a, b)
        (g,) = torch.autograd.grad(torch.sum(c), a, retain_graph=True)
