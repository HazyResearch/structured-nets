import torch
from torch.autograd import Variable
import time

def krylov(fn, v, n):
    cols = [v]
    for _ in range(n - 1):
        v = fn(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)

def krylov_recon(params, G, H, fn_A, fn_B_T):
	W1 = Variable(torch.zeros(params.layer_size, params.layer_size).cuda())

	for i in range(params.r):
		K_A = krylov(fn_A, G[:, i], params.layer_size)
		
		K_B = krylov(fn_B_T, H[:, i], params.layer_size).t()

		prod = torch.matmul(K_A, K_B).cuda()
		#print('W1: ', W1)
		#print('prod: ', prod)
		W1 += prod

	return W1
