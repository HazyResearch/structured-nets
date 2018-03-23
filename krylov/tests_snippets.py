import os
os.environ[‘MKL_NUM_THREADS’] = ‘1’

np.random.seed(0)


n, m = 2, 1
A = np.array([[0,0],[1,0]])
u = np.array([1,1])
v = np.array([1,1])
print(resolvent_bilinear(A,v,u,n))
# ans: [2 1], [1, 1], [1 1], [0 1]


n, m = 4, 2
A = np.diag(np.arange(1,4),-1)
u = np.ones(4)
v = np.ones(4)
print(resolvent_bilinear(A,v,u,4))
print(krylov_mult(A,v,u,4))
print(krylov_mult_slow(A,v,u,4))
print(krylov_mult_slow_faster(A,v,u,4))
print(resolvent_bilinear_flattened(A,v,u,4,2))
# ans: [4 6 8 6], [1 1 2 6], [1 3 6 6], [0 0 0 6]


m = 12
n = 1<<m
A = np.diag(np.random.random(n-1), -1)
u = np.random.random(n)
v = np.random.random(n)
k1 = krylov_mult_slow(A,v,u,n)
k1_allocated = krylov_mult_slow_allocated(A,v,u,n)
k11 = krylov_mult_slow_faster(A,v,u,n)
k2 = krylov_mult(A,v,u,n)
k3 = resolvent_bilinear_flattened(A, v, u, n, m, stack_fft=False)
k3b = resolvent_bilinear_flattened(A, v, u, n, m, stack_fft=False)
print(np.linalg.norm(k1-k11))
print(np.linalg.norm(k1-k2))
print(np.linalg.norm(k1-k3))
print(np.linalg.norm(k1-k3b))
