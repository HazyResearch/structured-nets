import torch
from scipy.linalg import circulant
import numpy as np
from .complex_utils import complex_mult

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Multiplies circulant matrix with first column c by x
def circulant_multiply(c, x, n):
    return torch.irfft(complex_mult(torch.rfft(c, 1), torch.rfft(x, 1)), 1, signal_sizes=(n, ))

def test_circulant_multiply(n):
    c = np.random.random(n)
    x = np.random.random(n)
    C = circulant(c)
    slow = np.dot(C,x)
    c = torch.tensor(c, dtype=torch.float, device=device)
    x = torch.tensor(x, dtype=torch.float, device=device)
    fast = circulant_multiply(c, x, n)
    print('Error compared to slow multiply: ', np.linalg.norm(slow-fast))

# TODO: move test into subpackage
if __name__ == '__main__':
    test_circulant_multiply(100)
