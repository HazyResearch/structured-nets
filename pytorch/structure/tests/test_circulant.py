import torch
from mle.structure.circulant import circulant_multiply
from scipy.linalg import circulant

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def test_circulant_multiply():
    n = 100
    c = torch.rand(n, device=device)
    x = torch.rand((3, n), device=device)
    C = torch.tensor(
        circulant(c.detach().cpu().numpy()), dtype=c.dtype, device=c.device
    )
    slow = x @ C.t()
    fast = circulant_multiply(c, x)

    torch.testing.assert_close(slow, fast)
