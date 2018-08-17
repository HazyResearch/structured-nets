import torch
from scipy.linalg import circulant
from .complex_utils import complex_mult

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def circulant_multiply(c, x):
    """ Multiply circulant matrix with first column c by x
    Parameters:
        c: (n, )
        x: (batch_size, n) or (n, )
    Return:
        prod: (batch_size, n) or (n, )
    """
    return torch.irfft(complex_mult(torch.rfft(c, 1), torch.rfft(x, 1)), 1, signal_sizes=(c.shape[-1], ))

def test_circulant_multiply(n):
    c = torch.rand(n, device=device)
    x = torch.rand((3, n), device=device)
    C = torch.tensor(circulant(c.detach().cpu().numpy()), dtype=c.dtype, device=c.device)
    slow = x @ C.t()
    fast = circulant_multiply(c, x)
    print('Error compared to slow multiply: ', (slow - fast).abs().max().item())

# TODO: move test into subpackage
if __name__ == '__main__':
    test_circulant_multiply(100)
