import numpy as np
import torch

from scipy.linalg import hadamard

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hadamard_transform(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    batch_size, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)


def test_hadamard_transform():
    m = 10
    n = 1 << m
    batch_size = 50
    u = torch.rand((batch_size, n), requires_grad=True, device=device)
    result = hadamard_transform(u)
    # Explicit construction from scipy
    H = torch.tensor(hadamard(n), dtype=torch.float, device=device)
    result_explicit = u @ H.t()
    print((result - result_explicit).abs().max().item())
    print((result - result_explicit).abs().mean().item())
