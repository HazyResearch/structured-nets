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
    """
    if n == 1:
        # don't know how write outer product in numpy
        return (np.array([[[ u[0]*v[0] ], [ u[0]*1 ]], [[ 1*v[0] ], [ 1*1 ]]]), np.array([1,-A[0,0]]))

    k = n//2
    # Let M00 = M[0:k, 0:k], M10 = M[k:n, 0:k], M11 = M[k:n,k:n]
    # i.e. M = [M00 0 ; M10 M11] (where M = I-Ax)
    # then M^{-1} = [M00^{-1} 0 ; -M11^{-1} M_10^{-1} M_00^{-1}]
    S0, d0 = resolvent_bilinear(A[:k,:k], v[:k], u[:k], k)
    S1, d1 = resolvent_bilinear(A[k:,k:], v[k:], u[k:], n-k)

    # the part corresponding to bottom left corner is
    # -A[k, k-1]x * u_1^T M_11^{-1} e_1 * e_k^T M_00^{-1} v_0
    # or S1[:,1] * S0[1,:]
    L = np.array([[poly_mult(S1[0,1], S0[1,0]), poly_mult(S1[0,1], S0[1,1])], [poly_mult( S1[1,1], S0[1,0] ), poly_mult( S1[1,1], S0[1,1] )]])
    # print(L)
    L = A[k,k-1] * np.pad(L, ((0,0),(0,0),(1,0)), 'constant') # multiply by X
    # TODO: above padding should be able to be optimized away; when we allocate memory properly can store the coefficients directly in the right place
    # print(L)

    # clear denominators
    # S0 = np.array([[ poly_mult(s, d1) for s in r ] for r in S0])
    # S1 = np.array([[ poly_mult(s, d0) for s in r ] for r in S1])
    # print(S0)

    # really need to define poly matrix operations
    # S = np.array([[poly_add(S0[i,j],S1[i,j]) for j in range(2)] for i in range(2)])
    # S = np.array([[poly_add(S[i,j],L[i,j]) for j in range(2)] for i in range(2)])
    # L[0,0] = poly_add(L[0,0], poly_mult(S0[0,0], d1), n)
    # L[0,1] = poly_add(L[0,1], poly_mult(S0[0,1], d1), n)
    # L[0,0] = poly_add(L[0,0], poly_mult(S1[0,0], d0), n)
    # L[1,0] = poly_add(L[1,0], poly_mult(S1[1,0], d0), n)
    L[0,0] += poly_mult(S0[0,0], d1) + poly_mult(S1[0,0], d0)
    L[0,1] += poly_mult(S0[0,1], d1)
    L[1,0] += poly_mult(S1[1,0], d0)
    return (L, poly_mult(d0,d1))

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

def krylov_mult_slow_allocated(A, v, u, m):
    n = v.shape[0]
    assert A.shape == (n,n)
    d = np.diagonal(A, 0)
    subd = np.diagonal(A, -1)
    # Allocate memory at once to K
    K_T = np.empty((m, n))
    K_T[0] = v
    for i in range(1,m):
        K_T[i] = Amult(d, subd, K_T[i-1])
    return K_T @ u

def krylov_construct(A, v, m):
    n = v.shape[0]
    assert A.shape == (n,n)
    d = np.diagonal(A, 0)
    subd = np.diagonal(A, -1)

    K = np.zeros(shape=(m,n))
    K[0,:] = v
    for i in range(1,m):
        K[i,1:] = subd*K[i-1,:-1]
    return K

def krylov_mult_slow_faster(A, v, u, m):
    K = krylov_construct(A, v, m)
    return K @ u
