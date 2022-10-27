import torch
from mle.structure.fastfood import fastfood_multiply
from scipy.linalg import hadamard

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def _test_fastfood_multiply(n, batch_size):
    S = torch.randn(n, device=device)
    G = torch.randn(n, device=device)
    B = torch.randn(n, device=device)
    P = torch.randperm(n, device=device)
    x = torch.randn(batch_size, n, device=device)
    H = torch.tensor(hadamard(n), dtype=torch.float, device=device)
    HBx = (H @ (B * x).T).T
    PHBx = HBx[:, P]
    HGPHBx = (H @ (G * PHBx).T).T
    output_explicit = S * HGPHBx

    S = torch.tensor(S, dtype=torch.float, device=device)
    G = torch.tensor(G, dtype=torch.float, device=device)
    B = torch.tensor(B, dtype=torch.float, device=device)
    P = torch.tensor(P, dtype=torch.long, device=device)
    x = torch.tensor(x, dtype=torch.float, device=device)

    output = fastfood_multiply(S, G, B, P, x)

    torch.testing.assert_close(output_explicit, output, rtol=1e-3, atol=1e-3)


def test_fastfood_multiply():
    _test_fastfood_multiply(128, 50)
