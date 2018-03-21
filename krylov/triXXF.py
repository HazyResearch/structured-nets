
import numpy as np
import scipy.fftpack as fft
from scipy import signal

# should create a poly class later
p1 = np.full(5, 2)
p2 = np.full(10, 3)

def poly_add(p1, p2, n):
    """p1,p2 of degree exactly n-1"""
    # TODO: change these to equals
    assert p1.shape == (n,)
    assert p2.shape == (n,)
    # n = np.maximum(p1.shape[0], p2.shape[0])
    # q1 = np.pad(p1, (0,n-p1.shape[0]), 'constant')
    # q2 = np.pad(p2, (0,n-p2.shape[0]), 'constant')
    # print(q1)
    # print(q2)
    return p1+p2

def poly_mult_slow(p1, p2):
    d1 = p1.shape[0] - 1
    d2 = p2.shape[0] - 1
    n = d1 + d2
    prod = np.zeros(n+1)
    for i in range(d1+1):
        for j in range(d2+1):
            prod[i+j] += p1[i]*p2[j]
    return prod

def poly_mult_fft(p1, p2):
    d1 = p1.shape[0] - 1
    d2 = p2.shape[0] - 1
    # if d1 < 0:
    #     p1 = np.array([0])
    #     d1 = 0
    # if d2 < 0:
    #     p2 = np.array([0])
    #     d2 = 0
    # n = d1 + d2

    # numpy fft
    # f1 = np.fft.rfft(p1, n+1)
    # f2 = np.fft.rfft(p2, n+1)
    # prod = np.fft.irfft(f1*f2, n+1)

    # scipy fft (currently has bug because it stores output of rfft differently)
    f1 = fft.rfft(p1, n+1)
    f2 = fft.rfft(p2, n+1)
    prod = fft.irfft(f1*f2, n+1)

    # prod = signal.convolve(p1, p2, method='fft')

    return prod

# define an alias for easy testing
def poly_mult(p1, p2):
    # return poly_mult_slow(p1, p2)
    d1 = p1.shape[0] - 1
    d2 = p2.shape[0] - 1
    n = d1 + d2
    # q1 = np.pad(p1, (0,d2), 'constant')
    # q2 = np.pad(p2, (0,d1), 'constant')
    # assert q1.shape[0] == n+1
    # assert q2.shape[0] == n+1
    if n >= 128:
        prod = signal.fftconvolve(p1, p2, mode='full')
    else:
        prod = np.convolve(p1, p2)
    # prod = np.convolve(p1, p2)
    # if prod.shape[0] != n+1:
    #     print(d1, d2, p1.shape, p2.shape, prod.shape)
    #     assert false
    # assert prod.shape[0] == n+1

    return prod

def poly_inv(p, n):
    """
    invert p mod x^n
    """
    assert n >= 1
    if n == 1:
        return np.array([1 / p[0]])

    # represent p = p_low + x^k p_high, and its inverse q similarly
    d = p.shape[0]
    k = (n+1)//2

    # invert the lower order terms
    q_low = poly_inv(p[:min(d,k)], k)
    # print(q_low)

    # since 2k >= n, p q_l + x^k p_l q_h = 1 (mod x^n)
    # so p_l q_h = (1 - p q_l)/x^k  (mod x^{n-k})
    r = poly_mult(p, q_low)
    r[0] -= 1
    # assert np.all(r[:min(r.shape[0],k)] == 0)
    # but we know p_l^{-1} mod x^{n-k} since we already know it mod x^k
    q_high = poly_mult(-r[k:min(r.shape[0],n)], q_low)

    # q_low = np.pad(q_low, (0,k-q_low.shape[0]), 'constant')
    q = np.concatenate((q_low, q_high))[:n]
    # q = np.trim_zeros(q, 'b')
    return q



def resolvent_bilinear(A, v, u, n):
    """
    Compute [u e_n]^T * (I-Ax)^{-1} * [v e_1]
    (2x2 matrix of rational fractions)
    output: array of shape (2, 2, n), array shape (n)
    (numerator, denominator)

    invariants:
        numerator has degree n-1
        denominator degree n

    # let x_{m-1}, x_{m-2}, ... x_{m-d} denote the branch
    #     i.e. this computes the answer for indices i = with bit pattern matching the above
    #     (since this fixes the higher-order bits, this is a contiguous range of u,v)
    # returns arrays indexed by bits y_0, ..., y_{m-d-1}, z_0, z_1
    # where n = 2^m
    """
    if n == 1: # leaf: branch x_0, \dots, x_{m-1}
        # don't know how write outer product in numpy
        return (np.array([[[ u[0]*v[0] ], [ u[0]*1 ]], [[ 1*v[0] ], [ 1*1 ]]]), np.array([1,-A[0,0]]))

    k = n//2
    # Let M00 = M[0:k, 0:k], M10 = M[k:n, 0:k], M11 = M[k:n,k:n]
    # i.e. M = [M00 0 ; M10 M11] (where M = I-Ax)
    # then M^{-1} = [M00^{-1} 0 ; -M11^{-1} M_10^{-1} M_00^{-1}]
    S0, d0 = resolvent_bilinear(A[:k,:k], v[:k], u[:k], k)   # subproblem [x_0, ..., x_d, 0]
    S1, d1 = resolvent_bilinear(A[k:,k:], v[k:], u[k:], n-k) # subproblem [x_0, ..., x_d, 1]

    # the part corresponding to bottom left corner is
    # -A[k, k-1]x * u_1^T M_11^{-1} e_1 * e_k^T M_00^{-1} v_0
    # or S1[:,1] * S0[1,:]
    L = np.array([[poly_mult(S1[0,1], S0[1,0]), poly_mult(S1[0,1], S0[1,1])], [poly_mult( S1[1,1], S0[1,0] ), poly_mult( S1[1,1], S0[1,1] )]])
    # this is x = [x_0, ..., x_d, 1], z = [0, 1]
    L = A[k,k-1] * np.pad(L, ((0,0),(0,0),(1,0)), 'constant') # multiply by X
    # TODO: above padding should be able to be optimized away; when we allocate memory properly can store the coefficients directly in the right place

    # clear denominators
    L[0,0] += poly_mult(S0[0,0], d1) + poly_mult(S1[0,0], d0)
    L[0,1] += poly_mult(S0[0,1], d1)
    L[1,0] += poly_mult(S1[1,0], d0)
    return (L, poly_mult(d0,d1))

# TODO: put this as subfunction of other

# this is specialized to subdiagonal for now
def resolvent_bilinear_flattened_(subd, v, u, m, d, S):
    # pass at depth d computes 4 arrays:
    # each array is length n, indexed by x_{m-1}, ..., x_{m-d}, y_{m-d-1}, ..., y_0
    # for convenience, store as x_{m-d}, ..., x_{m-1}, y_{m-d-1}, ..., y_0

    assert d < m # assume leaf pass done in main function

    S_00, S_01, S_10, S_11 = S # answers to previous layer: indexed by x_{m-d-1}, x_{m-d}, ..., x_{m-1}, y_{m-d-2}, ..., y_0
    # these are the same as the S0[0,0],S1[0,0] above
    assert S_00.shape == (1<<(d+1), 1<<(m-d-1))
    n1, n2 = 1<<d, 1<<(m-d)

    # TODO: time the following: hcat a big dense matrix with a 0 matrix, vs. np.zeros followed by assignment
    # %timeit np.hstack((x, np.zeros((32,32))))   : 9.16 µs ± 221 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    # %timeit y = np.zeros((32,64)); y[:,:32] = x : 3.63 µs ± 573 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    # S0_00, S0_01, S0_00, S0_01, S1_00, S1_01, S1_00, S1_01 = np.zeros((8,n))
    # S0_00[:n/2], S1_00[:n/2] = S_00[:n/2], S_00[n/2:] # alternative: numpy.split(S, 2)
    # S0_01[:n/2], S1_01[:n/2] = S_01[:n/2], S_01[n/2:]
    # S0_10[:n/2], S1_10[:n/2] = S_10[:n/2], S_10[n/2:]
    # S0_11[:n/2], S1_11[:n/2] = S_11[:n/2], S_11[n/2:]
    S0_10, S0_11, S1_01, S1_11 = np.zeros((4,n1,n2))
    S0_10[:,:n2//2] = S_10[:n1,:]
    S1_01[:,:n2//2] = S_01[n1:,:]
    S0_11[:,:n2//2] = S_11[:n1,:]
    S1_11[:,:n2//2] = S_11[n1:,:]

    # polynomial multiplications
    S0_10_ = np.fft.rfft(S0_10)
    S0_11_ = np.fft.rfft(S0_11)
    S1_01_ = np.fft.rfft(S1_01)
    S1_11_ = np.fft.rfft(S1_11)

    # subproblem for branch x_{m-d}, ..., x_{m-1} is A[\overline{x_{m-1}...x_{m-d}} + 2^{m-d-1}]
    A_subd = subd[n1:n1*2, np.newaxis]
    T_00 = A_subd * np.fft.irfft(S1_01_ * S0_10_)
    T_01 = A_subd * np.fft.irfft(S1_01_ * S0_11_)
    T_10 = A_subd * np.fft.irfft(S1_11_ * S0_10_)
    T_11 = A_subd * np.fft.irfft(S1_11_ * S0_11_)

    # polynomial additions
    T_00[:,n2//2:] += S_00[:n1,:]
    T_00[:,n2//2:] += S_00[n1:,:]
    T_01[:,n2//2:] += S_01[:n1,:]
    T_10[:,n2//2:] += S_10[n1:,:]

    return (T_00, T_01, T_10, T_11)


def idx_bitflip(x, n, m):
    assert n == 1<<m # power of 2 for now
    x_ = x.reshape([2]*m)
    x_bf_ = np.empty(shape=[2]*m)
    for i in itertools.product(*([[0,1]]*m)):
        x_bf_[i[::-1]] = x_[i]
    x_bf = x_bf_.reshape(n)
    return x_bf

def resolvent_bilinear_flattened(A, v, u, n, m):
    assert n == 1<<m # power of 2 for now

    # assume A is subdiagonal for now
    subd = np.concatenate(([0],np.diagonal(A, -1)))
    subd = idx_bitflip(subd, n, m)

    # reshape u,v to be indexed consistently with the above
    # i.e. bit flip their indices
    u_bf = idx_bitflip(u, n, m).reshape((n,1)) # tri says use [:,np.newaxis]
    v_bf = idx_bitflip(v, n, m).reshape((n,1))

    S = (u_bf*v_bf, u_bf, v_bf, np.ones((n,1)))

    for d in range(m-1,-1,-1):
        S = resolvent_bilinear_flattened_(subd, v, u, m, d, S)

    # print(S[0], S[1], S[2], S[3])
    # return np.flip(S[0], axis=-1)
    return S[0].squeeze()[::-1]





def krylov_mult(A, v, u, m):
    """
    Compute the matrix-vector product Kry(A, v)^T * u
    A: R^{n \times n}, lower triangular and 2-banded
    u: R^n
    v: R^n
    m: output dimension (i.e. width of K)
    """

    n = v.shape[0]
    assert A.shape == (n,n)

    R, d = resolvent_bilinear(A,v,u,n)
    ans = poly_mult(R[0,0], poly_inv(d, m))
    return ans[:m]

def Amult(d, subd, v):
    ans = d*v
    ans[1:] += subd*v[:-1]
    return ans

def krylov_mult_slow(A, v, u, m):
    n = v.shape[0]
    assert A.shape == (n,n)
    cols = [v]
    d = np.diagonal(A, 0)
    subd = np.diagonal(A, -1)
    for i in range(1,m):
        cols.append(Amult(d, subd, cols[-1]))
    K = np.stack(cols, axis=1)
    return K.T @ u

def krylov_mult_slow_faster(A, v, u, m):
    n = v.shape[0]
    assert A.shape == (n,n)
    d = np.diagonal(A, 0)
    subd = np.diagonal(A, -1)

    K = np.zeros(shape=(m,n))
    K[0,:] = v
    for i in range(1,m):
        K[i,1:] = subd*K[i-1,:-1]
    return K @ u

np.random.seed(0)

A = np.array([[0,0],[1,0]])
u = np.array([1,1])
v = np.array([1,1])
print(resolvent_bilinear(A,v,u,2))
# ans: [2 1], [1, 1], [1 1], [0 1]

# A = np.array([[0,0,0,0],[1,0,0,0],[0,2,0,0],[0,0,3,0]])
# u = np.array([1,1,1,1])
# v = np.array([1,1,1,1])
# resolvent_bilinear(A,u,v,4)

n = 4
A = np.diag(np.arange(1,4),-1)
u = np.ones(4)
v = np.ones(4)
print(resolvent_bilinear(A,v,u,4))
print(krylov_mult(A,v,u,4))
print(krylov_mult_slow(A,v,u,4))
print(krylov_mult_slow_faster(A,v,u,4))
print(resolvent_bilinear_flattened(A,v,u,4,2))
# ans: [4 6 8 6], [1 1 2 6], [1 3 6 6], [0 0 0 6]

# A = np.array([[0,0,0,0],[1,0,0,0],[0,2,0,0],[0,0,3,0]])
# u = np.array([1,1,1,1])
# v = np.array([1,1,1,1])
# resolvent_bilinear(A,u,v,4)
m = 12
n = 1<<m
A = np.diag(np.random.random(n-1), -1)
u = np.random.random(n)
v = np.random.random(n)
k1 = krylov_mult_slow(A,v,u,n)
k11 = krylov_mult_slow_faster(A,v,u,n)
k2 = krylov_mult(A,v,u,n)
k3 = resolvent_bilinear_flattened(A, v, u, n, m)
print(np.max(np.abs(k1-k11)))
print(np.max(np.abs(k1-k2)))
print(np.max(np.abs(k1-k3)))
