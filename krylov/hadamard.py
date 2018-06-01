import numpy as np
import torch

use_hadamard_transform_cuda = True
try:
    import cuda_extension
except ImportError:
    print("CUDA version of Hadamard transform isn't installed. Will use Pytorch's version, which is much slower.")
    use_hadamard_transform_cuda = False

from scipy.linalg import hadamard

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hadamard_transform_torch(u, normalize=False):
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


class HadamardTransformCuda(torch.autograd.Function):
    '''The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))
    '''
    @staticmethod
    def forward(ctx, u):
        return cuda_extension.hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return HadamardTransformCuda.apply(grad)


def hadamard_transform_cuda(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    _, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    output = HadamardTransformCuda.apply(u)
    return output / 2**(m / 2) if normalize else output


def test_hadamard_transform():
    m = 15
    n = 1 << m
    batch_size = 50
    u = torch.rand((batch_size, n), requires_grad=True, device=device)
    result_cuda = hadamard_transform_cuda(u)
    grad_cuda, = torch.autograd.grad(result_cuda.sum(), u, retain_graph=True)
    result_torch = hadamard_transform_torch(u)
    grad_torch, = torch.autograd.grad(result_torch.sum(), u, retain_graph=True)
    # Explicit construction from scipy
    H = torch.tensor(hadamard(n), dtype=torch.float, device=device)
    result_explicit = u @ H.t()
    print((result_cuda - result_explicit).abs().max().item())
    print((result_cuda - result_explicit).abs().mean().item())
    print((result_torch - result_explicit).abs().max().item())
    print((result_torch - result_explicit).abs().mean().item())
    print((grad_cuda - grad_torch).abs().max().item())
    print((grad_cuda - grad_torch).abs().mean().item())


hadamard_transform = hadamard_transform_cuda if use_hadamard_transform_cuda else hadamard_transform_torch
