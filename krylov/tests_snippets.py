import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import numpy as np
from triXXF import *
from triextrafat import *

np.random.seed(0)


# n, m = 2, 1
# A = np.array([[0,0],[1,0]])
# u = np.array([1,1])
# v = np.array([1,1])
# print(resolvent_bilinear(A,v,u,n))
# ans: [2 1], [1, 1], [1 1], [0 1]


# n, m = 4, 2
# A = np.diag(np.arange(1,4),-1)
# u = np.ones(4)
# v = np.ones(4)
# print(resolvent_bilinear(A,v,u,4))
# print(krylov_mult(A,v,u,4))
# print(krylov_mult_slow(A,v,u,4))
# print(krylov_mult_slow_faster(A,v,u,4))
# print(resolvent_bilinear_flattened(A,v,u,4,2))
# ans: [4 6 8 6], [1 1 2 6], [1 3 6 6], [0 0 0 6]


m = 14
n = 1<<m
subdiag = np.random.random(n-1)
A = np.diag(subdiag, -1)
u = np.random.random(n)
v = np.random.random(n)
# k1 = krylov_mult_slow(A,v,u,n)
# k1_allocated = krylov_mult_slow_allocated(A,v,u,n)
# k11 = krylov_mult_slow_faster(A,v,u,n)
# k2 = krylov_mult(A,v,u,n)
resolvent_bilinear_flattened = create(n, m, lib='fftw')
krylov_transpose_multiply = KrylovTransposeMultiply(n)
k3 = resolvent_bilinear_flattened(A, v, u, n, m)
k3_nobf = krylov_transpose_multiply(subdiag, v, u)
np.allclose(k3, k3_nobf)
[resolvent_bilinear_flattened(A, v, u, n, m) for i in range(100)]
# print(np.linalg.norm(k1-k11))
# print(np.linalg.norm(k1-k2))
# print(np.linalg.norm(k1-k3))
# print(np.linalg.norm(k1-k3b))

# Test non-transpose multiply
m = 14
n = 1 << m
subdiag = np.random.random(n-1)
A = np.diag(subdiag, -1)
u = np.random.random(n)
v = np.random.random(n)
krylov_multiply = KrylovMultiply(n)
result = krylov_construct(A, v, n).T @ u
result1 = krylov_multiply(subdiag, v, u)
np.allclose(result, result1)

# Test batch transpose multiply
m = 12
n = 1<<m
batch_size = 100
subdiag = np.random.random(n-1)
A = np.diag(subdiag, -1)
u = np.random.random((n, batch_size))
v = np.random.random(n)
krylov_transpose_multiply = KrylovTransposeMultiply(n, batch_size)
result1 = krylov_transpose_multiply(subdiag, v, u)
resolvent_bilinear_flattened = create(n, m, lib='fftw')
result = resolvent_bilinear_flattened(A, v, u[:, 10], n, m)
np.allclose(result1[:, 10], result)
K = krylov_construct(A, v, n)
result2 = K @ u
np.allclose(result1, result2)

# Test batch non-transpose multiply
m = 14
n = 1<<m
batch_size = 100
subdiag = np.random.random(n-1)
A = np.diag(subdiag, -1)
u = np.random.random((n, batch_size))
v = np.random.random(n)
krylov_multiply = KrylovMultiply(n, batch_size)
result1 = krylov_multiply(subdiag, v, u)
K = krylov_construct(A, v, n)
result2 = K.T @ u
np.allclose(result1, result2)

# Test batch transpose multiply with rank >= 2
m = 14
n = 1<<m
batch_size = 100
rank = 3
subdiag = np.random.random(n-1)
A = np.diag(subdiag, -1)
u = np.random.random((batch_size, n))
v = np.random.random((rank, n))
krylov_transpose_multiply = KrylovTransposeMultiply(n, batch_size, rank)
result1 = krylov_transpose_multiply(subdiag, v, u)
resolvent_bilinear_flattened = create(n, m, lib='fftw')
result = resolvent_bilinear_flattened(A, v[0], u[10], n, m)
np.allclose(result1[10, 0], result)
Ks = [krylov_construct(A, v[i], n) for i in range(rank)]
result2 = np.stack([u @ K.T for K in Ks]).swapaxes(0, 1).squeeze()
np.allclose(result1, result2)

# Test batch non-transpose multiply with rank >= 2
m = 14
n = 1<<m
batch_size = 100
# batch_size = 1
rank = 3
# rank = 2
subdiag = np.random.random(n-1)
A = np.diag(subdiag, -1)
u = np.random.random((batch_size, n))
v = np.random.random((rank, n))
krylov_multiply = KrylovMultiply(n, batch_size, rank)
result1 = krylov_multiply(subdiag, v, u)
Ks = [krylov_construct(A, v[i], n) for i in range(rank)]
result2 = np.stack([u @ K for K in Ks]).swapaxes(0, 1).squeeze()
np.allclose(result1, result2)
