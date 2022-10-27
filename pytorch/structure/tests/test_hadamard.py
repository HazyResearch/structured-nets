import scipy.linalg
import torch
from mle.structure import hadamard
from mle.structure.hadamard import hadamard_transform_cuda, hadamard_transform_torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def test_hadamard_torch():
    u = torch.tensor([1, 1])
    H = torch.tensor([[1, 1], [1, -1]])
    res1 = H @ u
    res2 = hadamard.hadamard_transform_torch(u[None])[0]
    torch.testing.assert_allclose(res1, res2)


def test_hadamard_cuda():
    B, D = 10, 2**6
    X = torch.randn(B, D)
    H1 = hadamard.hadamard_transform_torch(X)
    H2 = hadamard.hadamard_transform_cuda(X.cuda()).cpu()
    torch.testing.assert_close(H1, H2)


def test_hadamard_transform():
    batch_size = 50
    for m in range(10):
        n = 1 << m
        # We transform +/- 1s, since transforming randn's give issues with accuracy for larger m...
        u = torch.randint(
            2, (batch_size, n), requires_grad=True, device=device, dtype=torch.float
        )

        result_cuda = hadamard_transform_cuda(u)
        result_torch = hadamard_transform_torch(u)

        # Explicit construction from scipy
        H = torch.tensor(scipy.linalg.hadamard(n), dtype=torch.float, device=device)
        result_explicit = u @ H

        torch.testing.assert_close(result_torch, result_explicit)
        torch.testing.assert_close(result_cuda, result_explicit)

        (grad_cuda,) = torch.autograd.grad(result_cuda.sum(), u, retain_graph=True)
        (grad_torch,) = torch.autograd.grad(result_torch.sum(), u, retain_graph=True)

        torch.testing.assert_close(grad_cuda, grad_torch)
