from .hadamard import hadamard_transform
import torch
import numpy as np
from scipy.linalg import hadamard

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# S,G,B: diagonal
# P: permutation
# x: batch_size x n_features
def fastfood_multiply(S,G,B,P,x):
    HBx = hadamard_transform(B*x)
    PHBx = HBx[:, P]
    HGPHBx = hadamard_transform(G*PHBx)
    return S*HGPHBx

def test_fastfood_multiply(n, batch_size):
    S = np.random.randn(n)
    G = np.random.randn(n)
    B = np.random.randn(n)
    P = np.random.permutation(n)
    x = np.random.randn(batch_size,n)
    H = hadamard(n)
    HBx = np.dot(H,(B*x).T).T
    PHBx = HBx[:,P]
    HGPHBx = np.dot(H,(G*PHBx).T).T
    output_explicit = S*HGPHBx

    S = torch.tensor(S, dtype=torch.float, device=device)
    G = torch.tensor(G, dtype=torch.float, device=device)
    B = torch.tensor(B, dtype=torch.float, device=device)
    P = torch.tensor(P, dtype=torch.long, device=device)
    x = torch.tensor(x, dtype=torch.float, device=device)

    output = fastfood_multiply(S,G,B,P,x)

    print(np.linalg.norm(output_explicit - output))

# TODO: move test into subpackage
if __name__ == '__main__':
    test_fastfood_multiply(128,50)
