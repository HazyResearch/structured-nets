import numpy as np
import scipy.fftpack as fft
import itertools
from scipy import signal

import pyfftw


# define fft calls
def _plan_ffts(in_shape, lib='numpy'):
    out_shape = in_shape[:-1] + (in_shape[-1]//2 + 1,)
    if lib == 'numpy':
        x_for = np.empty(shape=in_shape)
        # y_for = np.empty(shape=out_shape)
        fft = lambda: np.fft.rfft(x_for)

        y_bak = np.empty(shape=out_shape, dtype='complex128')
        ifft = lambda: np.fft.irfft(y_bak)
        return ((x_for, fft), (y_bak, ifft))
    if lib == 'scipy':
        pass
    if lib == 'fftw':
        out_shape = in_shape[:-1] + (in_shape[-1]//2 + 1,)
        x_for = pyfftw.empty_aligned(in_shape, dtype='float64')
        y_for = pyfftw.empty_aligned(out_shape, dtype='complex128')
        fft_for = pyfftw.FFTW(x_for, y_for, direction='FFTW_FORWARD', flags=['FFTW_EXHAUSTIVE', 'FFTW_DESTROY_INPUT'])

        x_bak = pyfftw.empty_aligned(in_shape, dtype='float64')
        y_bak = pyfftw.empty_aligned(out_shape, dtype='complex128')
        fft_bak = pyfftw.FFTW(y_bak, x_bak, direction='FFTW_BACKWARD', flags=['FFTW_EXHAUSTIVE', 'FFTW_DESTROY_INPUT'])
        return ((x_for, fft_for), (y_bak, fft_bak))


def plan_ffts(m, lib='numpy'):
    fft_plans = [None] * m
    fft_mem_for = [None] * m
    fft_mem_bak = [None] * m
    for d in range(m-1,-1,-1):
        n1, n2 = 1<<d, 1<<(m-d)
        in_shape  = (4,n1,n2)
        # out_shape = (4,n1,n2//2+1)
        # fft_mem_for[d], fft_mem_bak[d], fft_plans[d] = _plan_ffts(in_shape, lib)
        fft_plans[d] = _plan_ffts(in_shape, lib)

        # fft_plans[d] = fft_wrapper(shape, fft_plans[d+1], lib)
    return fft_plans



# TODO: put this as subfunction of main function

# this is specialized to subdiagonal for now
def _resolvent_bilinear_flattened(fft_plans, subd, v, u, m, d, S):
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
    ((S_, fft), (T_, ifft)) = fft_plans[d]
    S_[:] = 0
    # S_ = np.zeros((4,n1,n2))
    S0_10, S0_11, S1_01, S1_11 = S_
    S0_10[:,:n2//2] = S_10[:n1,:]
    S1_01[:,:n2//2] = S_01[n1:,:]
    S0_11[:,:n2//2] = S_11[:n1,:]
    S1_11[:,:n2//2] = S_11[n1:,:]

    # polynomial multiplications
    S0_10_f, S0_11_f, S1_01_f, S1_11_f = fft()
    # S0_10_f, S0_11_f, S1_01_f, S1_11_f = np.fft.rfft(S_)

    # subproblem for branch x_{m-d}, ..., x_{m-1} is A[\overline{x_{m-1}...x_{m-d}} + 2^{m-d-1}]
    A_subd = subd[n1:n1*2, np.newaxis]
    # T_[:] = np.stack((S1_01_f * S0_10_f, S1_01_f * S0_11_f, S1_11_f * S0_10_f, S1_11_f * S0_11_f))
    T_[0] = S1_01_f * S0_10_f
    T_[1] = S1_01_f * S0_11_f
    T_[2] = S1_11_f * S0_10_f
    T_[3] = S1_11_f * S0_11_f
    T__ = ifft()
    T_00, T_01, T_10, T_11 = A_subd * T__
    # T_00, T_01, T_10, T_11 = A_subd * np.fft.irfft(np.stack(
    #     (S1_01_f * S0_10_f,
    #      S1_01_f * S0_11_f,
    #      S1_11_f * S0_10_f,
    #      S1_11_f * S0_11_f)
    # ))

    # polynomial additions
    T_00[:,n2//2:] += S_00[:n1,:]
    T_00[:,n2//2:] += S_00[n1:,:]
    T_01[:,n2//2:] += S_01[:n1,:]
    T_10[:,n2//2:] += S_10[n1:,:]

    return (T_00, T_01, T_10, T_11)

# another bit reversal algorithm that's asymptotically faster:
# http://www.lite.etsii.urjc.es/manuel.rubiosanchez/papers/Bit-Reversal-Article.pdf

def bitreversal_fat(x, n, m):
    """ Compute the bit reversal permutation """
    assert n == 1<<m # power of 2 for now
    x_ = x.reshape([2]*m)
    x_bf_ = np.empty(shape=[2]*m)
    for i in itertools.product(*([[0,1]]*m)):
        x_bf_[i[::-1]] = x_[i]
    x_bf = x_bf_.reshape(n)
    return x_bf

def bitreversal(x, n, m):
    """ faster version in numpy """
    assert n == 1<<m
    n1, n2 = n, 1
    x_ = x.reshape((n1,n2))
    for i in range(m):
        n1 //= 2
        n2 *= 2
        x_ = np.hstack((x_[:n1,:], x_[n1:,:]))
    return x_.squeeze()


# call with:
# resolvent_bilinear_flattened = create(n, m, 'numpy')
def create(n, m, lib='numpy'):
    fft_plans = plan_ffts(m, lib)

    def resolvent_bilinear_flattened(A, v, u, n, m):
        assert n == 1<<m # power of 2 for now

        # assume A is subdiagonal for now
        subd = np.empty((n,))
        subd[1:] = np.diagonal(A, -1)
        subd = bitreversal(subd, n, m)

        # reshape u,v to be indexed consistently with the above
        # i.e. bit flip their indices
        u_bf = bitreversal(u, n, m).reshape((n,1)) # tri says use [:,np.newaxis]
        v_bf = bitreversal(v, n, m).reshape((n,1))

        S = (u_bf*v_bf, u_bf, v_bf, np.ones((n,1)))

        # fft_plans = plan_ffts(m, lib)
        for d in range(m-1,-1,-1):
            S = _resolvent_bilinear_flattened(fft_plans, subd, v, u, m, d, S)

        # print(S[0], S[1], S[2], S[3])
        # return np.flip(S[0], axis=-1)
        return S[0].squeeze()[::-1]

    return resolvent_bilinear_flattened
