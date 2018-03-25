import numpy as np
import scipy.fftpack as fft
import itertools
from scipy import signal

import pyfftw


# define fft calls
def _plan_ffts(in_shape, lib='numpy'):
    out_shape = in_shape[:-1] + (in_shape[-1]//2 + 1,)
    if lib == 'numpy':
        x_for = np.zeros(shape=in_shape)
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
        fft_for = pyfftw.FFTW(x_for, y_for, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']) # don't destroy input so 0s are preserved
        x_for[:] = 0

        x_bak = pyfftw.empty_aligned(in_shape, dtype='float64')
        y_bak = pyfftw.empty_aligned(out_shape, dtype='complex128')
        fft_bak = pyfftw.FFTW(y_bak, x_bak, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
        return ((x_for, fft_for), (y_bak, fft_bak))


def plan_ffts(m, lib='numpy'):
    fft_plans = [None] * m
    for d in range(m-1,-1,-1):
        n1, n2 = 1<<d, 1<<(m-d)
        in_shape  = (4,n1,n2)
        fft_plans[d] = _plan_ffts(in_shape, lib)
    return fft_plans



# TODO: put this as subfunction of main function

# this is specialized to subdiagonal for now
# @profile
def _resolvent_bilinear_flattened(fft_plans, subd, m, d, S):
    # pass at depth d computes 4 arrays:
    # each array is length n, indexed by x_{m-1}, ..., x_{m-d}, y_{m-d-1}, ..., y_0
    # for convenience, store as x_{m-d}, ..., x_{m-1}, y_{m-d-1}, ..., y_0 (index is bit-reversed)

    # assert d < m # assume leaf pass done in main function

    S_00, S_01, S_10, S_11 = S # answers to previous layer: indexed by x_{m-d-1}, x_{m-d}, ..., x_{m-1}, y_{m-d-2}, ..., y_0
    # these are the same as the S0[0,0],S1[0,0] in the recursive version

    # assert S_00.shape == (1<<(d+1), 1<<(m-d-1))
    n1, n2 = 1<<d, 1<<(m-d-1) # input shape 2n1 x n2, output shape n1 x 2n2

    ((S_, fft), (T_, ifft)) = fft_plans[d]
    S0_10, S0_11, S1_01, S1_11 = S_ ## pass
    S0_10[:,:n2] = S_10[:n1,:]
    S1_01[:,:n2] = S_01[n1:,:]
    S0_11[:,:n2] = S_11[:n1,:]
    S1_11[:,:n2] = S_11[n1:,:] ## dS_11[...] = dS1_11[...]

    # polynomial multiplications
    S0_10_f, S0_11_f, S1_01_f, S1_11_f = fft() ## dS_ = fft(dS*_**_f)

    # subproblem for branch x_{m-d}, ..., x_{m-1} is A[\overline{x_{m-1}...x_{m-d}} + 2^{m-d-1}]
    T_[0] = S1_01_f * S0_10_f
    T_[1] = S1_01_f * S0_11_f
    T_[2] = S1_11_f * S0_10_f
    T_[3] = S1_11_f * S0_11_f  ## dS1_01_f += dT_[0] * S0_10_f; dS0_10_f += dT_[0] * S1_01_f
    ## note that the S*_**_f are the only things that need to be stored in t he forward pass
    ## also note that there is an optimization here; should only need half

    T__ = ifft() ## dT_ = ifft(dT__) (because DFT matrix symmetric)
    T__ *= subd[n1:n1*2, np.newaxis] ## dT__ *= subd[...]
    ## for learning A, should get somethiign like dsubd[...] = T__

    T_00, T_01, T_10, T_11 = T__

    # polynomial additions
    T_00[:,n2:] += S_00[:n1,:] ## dS_00[:n1,:] = T_00[:,n2:]
    T_00[:,n2:] += S_00[n1:,:]
    T_01[:,n2:] += S_01[:n1,:]
    T_10[:,n2:] += S_10[n1:,:]

    ## autodiff correspondences annotated in with '##'
    ## this function takes in S and outputs T;
    ## the backwards pass calls these lines in reverse,
    ## taking dT and outputting dS where ## dx := \partial{L}/\partial{x},
    ## (L is the final output of the entire algorithm)

    return (T_00, T_01, T_10, T_11)

def bitreversal_slow(x, n, m):
    """ Compute the bit reversal permutation """
    assert n == 1<<m # power of 2 for now
    x_ = x.reshape([2]*m)
    x_bf_ = np.empty(shape=[2]*m)
    for i in itertools.product(*([[0,1]]*m)):
        x_bf_[i[::-1]] = x_[i]
    x_bf = x_bf_.reshape(n)
    return x_bf

# note that this can be sped up by pre-allocating memory:
# %timeit np.hstack((x, np.zeros((32,32))))   : 9.16 µs ± 221 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# %timeit y = np.zeros((32,64)); y[:,:32] = x : 3.63 µs ± 573 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
def bitreversal_stack(x, n, m):
    """ faster version in numpy """
    assert n == 1<<m
    n1, n2 = n, 1
    x_ = x.reshape((n1,n2))
    for i in range(m):
        n1 //= 2
        n2 *= 2
        x_ = np.hstack((x_[:n1,:], x_[n1:,:]))
    return x_.squeeze()

# another bit reversal algorithm that's asymptotically faster:
# http://www.lite.etsii.urjc.es/manuel.rubiosanchez/papers/Bit-Reversal-Article.pdf
# we don't need to implement any fast ones because we only need to calculae the permutation once and then index into it


# call with:
# resolvent_bilinear_flattened = create(n, m, 'numpy')
def create(n, m, lib='numpy'):
    fft_plans = plan_ffts(m, lib)
    bf_perm = bitreversal_stack(np.arange(n), n, m)
    # Shorter versions but much slower. Maybe we don't care about speed because
    # this will done only once.
    # bf_perm_1 = np.array([int(np.binary_repr(i, width=m)[::-1], 2) for i in range(n)])
    # bf_perm_2 = np.array([int(f'{x:0{m}b}'[::-1], 2) for i in range(n)])
    # bf_perm_3 = np.array([int(bin(i + n)[:2:-1], 2) for i in range(n)])
    bitreversal = lambda x, n, m: x[bf_perm]

    # @profile
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

        for d in range(m)[::-1]:
            S = _resolvent_bilinear_flattened(fft_plans, subd, m, d, S)

        # return np.flip(S[0], axis=-1)
        return S[0].squeeze()[::-1]

    return resolvent_bilinear_flattened

class KrylovTransposeMultiply():
    """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
    """

    def __init__(self, n, lib='numpy'):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.fft_plans = plan_ffts(m, lib)

    def __call__(self, subdiag, v, u):
        """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
        We don't use bit reversal here.
        """
        n, m = self.n, self.m
        u, v = u[:, np.newaxis], v[:, np.newaxis]
        S = (u * v, u, v, np.ones((n, 1)))
        for d in range(m)[::-1]:
            n1, n2 = 1 << d, 1 << (m - d - 1)
            S_00, S_01, S_10, S_11 = S
            ((S_, fft), (T_, ifft)) = self.fft_plans[d]

            S0_10, S0_11, S1_01, S1_11 = S_ ## pass
            S0_10[:,:n2] = S_10[::2]
            S1_01[:,:n2] = S_01[1::2]
            S0_11[:,:n2] = S_11[::2]
            S1_11[:,:n2] = S_11[1::2] ## dS_11[...] = dS1_11[...]

            # polynomial multiplications
            S0_10_f, S0_11_f, S1_01_f, S1_11_f = fft() ## dS_ = fft(dS*_**_f)

            T_[0] = S1_01_f * S0_10_f
            T_[1] = S1_01_f * S0_11_f
            T_[2] = S1_11_f * S0_10_f
            T_[3] = S1_11_f * S0_11_f  ## dS1_01_f += dT_[0] * S0_10_f; dS0_10_f += dT_[0] * S1_01_f
            ## note that the S*_**_f are the only things that need to be stored in t he forward pass
            ## also note that there is an optimization here; should only need half

            T = ifft() ## dT_ = ifft(dT) (because DFT matrix symmetric)
            T *= subdiag[(n2 - 1)::(2 * n2), np.newaxis] ## dT *= subdiag[...]
            ## for learning A, should get something like dsubd[...] = T

            T_00, T_01, T_10, T_11 = T

            # polynomial additions
            T_00[:,n2:] += S_00[::2] ## dS_00[:n1,:] = T_00[:,n2:]
            T_00[:,n2:] += S_00[1::2]
            T_01[:,n2:] += S_01[::2]
            T_10[:,n2:] += S_10[1::2]

            ## autodiff correspondences annotated in with '##'
            ## this function takes in S and outputs T;
            ## the backwards pass calls these lines in reverse,
            ## taking dT and outputting dS where ## dx := \partial{L}/\partial{x},
            ## (L is the final output of the entire algorithm)

            S = T

        return S[0].squeeze()[::-1]


class KrylovMultiply():
    """Multiply Krylov(A, v) @ w when A is zero except on the subdiagonal.
    """

    def __init__(self, n, lib='numpy'):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.fft_plans = self.plan_ffts_forward_u_zero(m, lib)

    def _plan_ffts_forward_u_zero(self, in_shape, out_shape, lib='numpy'):
        if lib == 'numpy':
            x_for = np.zeros(shape=in_shape)
            fft = lambda: np.fft.rfft(x_for)

            y_bak = np.empty(shape=out_shape, dtype='complex128')
            ifft = lambda: np.fft.irfft(y_bak)
            return ((x_for, fft), (y_bak, ifft))
        if lib == 'scipy':
            pass
        if lib == 'fftw':
            out_shape_forward = in_shape[:-1] + (in_shape[-1]//2 + 1,)
            x_for = pyfftw.empty_aligned(in_shape, dtype='float64')
            y_for = pyfftw.empty_aligned(out_shape_forward, dtype='complex128')
            fft_for = pyfftw.FFTW(x_for, y_for, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']) # don't destroy input so 0s are preserved
            x_for[:] = 0

            in_shape_backward = out_shape[:-1] + ((out_shape[-1] - 1) * 2,)
            x_bak = pyfftw.empty_aligned(in_shape_backward, dtype='float64')
            y_bak = pyfftw.empty_aligned(out_shape, dtype='complex128')
            fft_bak = pyfftw.FFTW(y_bak, x_bak, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
            return ((x_for, fft_for), (y_bak, fft_bak))

    def plan_ffts_forward_u_zero(self, m, lib='numpy'):
        fft_plans = [None] * m
        for d in range(m-1,-1,-1):
            n1, n2 = 1<<d, 1<<(m-d)
            in_shape  = (3, n1, n2)
            out_shape = (2, n1, n2 // 2 + 1)
            fft_plans[d] = self._plan_ffts_forward_u_zero(in_shape, out_shape, lib)
        return fft_plans


    def forward(self, subdiag, v):
        """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
        Special case when u = 0 to save intermediate results relating to v for
        the backward pass.
        """
        n, m = self.n, self.m
        v = v[:, np.newaxis]
        S = (v, np.ones((n, 1)))
        for d in range(m)[::-1]:
            n1, n2 = 1 << d, 1 << (m - d - 1)
            S_10, S_11 = S
            ((S_, fft), (T_, ifft)) = self.fft_plans[d]

            S0_10, S0_11, S1_11 = S_ ## pass
            S0_10[:,:n2] = S_10[::2]
            S0_11[:,:n2] = S_11[::2]
            S1_11[:,:n2] = S_11[1::2] ## dS_11[...] = dS1_11[...]

            # polynomial multiplications
            S0_10_f, S0_11_f, S1_11_f = fft() ## dS_ = fft(dS*_**_f)

            T_[0] = S1_11_f * S0_10_f
            T_[1] = S1_11_f * S0_11_f  ## dS1_01_f += dT_[0] * S0_10_f; dS0_10_f += dT_[0] * S1_01_f
            ## note that the S*_**_f are the only things that need to be stored in t he forward pass
            ## also note that there is an optimization here; should only need half

            T = ifft() ## dT_ = ifft(dT) (because DFT matrix symmetric)
            T *= subdiag[(n2 - 1)::(2 * n2), np.newaxis] ## dT *= subdiag[...]
            ## for learning A, should get something like dsubd[...] = T

            T_10, T_11 = T

            # polynomial additions
            T_10[:,n2:] += S_10[1::2]

            ## autodiff correspondences annotated in with '##'
            ## this function takes in S and outputs T;
            ## the backwards pass calls these lines in reverse,
            ## taking dT and outputting dS where ## dx := \partial{L}/\partial{x},
            ## (L is the final output of the entire algorithm)

            S = (T_10, T_11)

    def __call__(self, subdiag, v, w):
        n, m = self.n, self.m
        self.forward(subdiag, v.reshape(n))
        v = v.reshape((n, 1))
        # We can ignore dT[2] and dT[3] because they start at zero and always stay zero.
        # We can check this by static analysis of the code or by math.
        dT = (w[::-1].reshape((1, n)), np.zeros((1, n)))

        for d in range(m):
            n1, n2 = 1 << d, 1 << (m - d - 1)
            ((S_, fft), (T_, ifft)) = self.fft_plans[d]
            dT_00, dT_01 = dT
            dS = np.zeros((2, 2 * n1, n2))
            dS_00, dS_01 = dS

            dS_00[::2] = dT_00[:, n2:]
            dS_00[1::2] = dT_00[:, n2:]
            dS_01[::2] = dT_01[:, n2:]
            dT *= subdiag[(n2 - 1)::(2 * n2), np.newaxis] ## dT *= subdiag[...]

            # Discard the negative frequencies since it's the complex conj of the positive frequencies
            dT_ = np.fft.ifft(dT)[:, :, :n2 + 1]
            dS1_01_f = np.zeros((n1, n2 + 1), dtype=dT_.dtype)
            S0_10_f, S0_11_f, S1_11_f = fft.output_array
            dS1_01_f += S0_10_f * dT_[0]
            dS1_01_f += S0_11_f * dT_[1]
            # This is inefficient. There's a way to do this with irfft but I'm to tired to think of it right now.
            dS1_01_f_temp = np.zeros((n1, 2 * n2), dtype=dT_.dtype)
            dS1_01_f_temp[:, :n2 + 1] = dS1_01_f
            dS1_01_f_temp[:, (2*n2 - 1):n2:-1] = np.conj(dS1_01_f_temp[:, 1:n2])

            dS_ = np.fft.fft(dS1_01_f_temp)
            assert np.allclose(dS_.imag, 0)
            dS_ = dS_.real
            dS1_01 = dS_
            dS_01[1::2] += dS1_01[:, :n2]

            dT = dS

        du = (dS[0] * v + dS[1]).squeeze()
        return du
