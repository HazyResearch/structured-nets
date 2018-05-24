import numpy as np
import matplotlib.pyplot as plt

# name = 'mnist_sd_r4'
names = ['bgrot_sd_r1', 'bgrot_sd_r4', 'bgrot_sd_r16']
# names = ['patch2_sd_r8', 'patch_sd_r8_best']
ranks = [1, 1, 4, 4, 16, 16]

for name in names:
# for name in [names[2]]:
    n = 1024
    # r = 4
    G = np.loadtxt(name+'_G')
    H = np.loadtxt(name+'_H')
    sd_A = np.loadtxt(name+'_subd_A')
    sd_B = np.loadtxt(name+'_subd_B')
    plt.figure()
    plt.imshow(sd_B.reshape(32,32), cmap='gray')
    plt.savefig(name+'.pdf', bbox_inches='tight')
    plt.clf()
    # plt.show()


# def krylov_construct(A, v, m):
#     n = v.shape[0]
#     assert A.shape == (n,n)
#     d = np.diagonal(A, 0)
#     subd = np.diagonal(A, -1)

#     K = np.zeros(shape=(m,n))
#     K[0,:] = v
#     for i in range(1,m):
#         K[i,1:] = subd*K[i-1,:-1]
#     return K

# A = np.diag(sd_A[:-1], -1)
# B = np.diag(sd_B[:-1], -1)
# M = sum([krylov_construct(A, G[i], n) @ krylov_construct(B, H[i], n).T for i in list(range(r))])
