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
        # Allocate memory up front
        S_f_storage = [np.empty((4, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        T_f_storage = [np.empty((4, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        S_storage = np.zeros((m, 4, n))
        T_storage = np.empty((m, 4, n))
        u, v = u.reshape(n, 1), v.reshape(n, 1)
        T_prev = (u * v, u, v, np.ones((n, 1)))
        for d in range(m)[::-1]:
            n1, n2 = 1 << d, 1 << (m - d - 1)
            S = S_storage[d].reshape((4, n1, 2 * n2))
            T = T_storage[d].reshape((4, n1, 2 * n2))
            ((_, fft), (T_f, ifft)) = self.fft_plans[d]

            S_00, S_01, S_10, S_11 = T_prev
            S0_10, S0_11, S1_01, S1_11 = S
            S0_10[:, :n2] = S_10[::2]
            S1_01[:, :n2] = S_01[1::2]
            S0_11[:, :n2] = S_11[::2]
            S1_11[:, :n2] = S_11[1::2]

            # polynomial multiplications
            S_f = S_f_storage[d]
            S0_10_f, S0_11_f, S1_01_f, S1_11_f = fft(S, output_array=S_f) ## dS_ = fft(dS*_**_f)
            T_f = T_f_storage[d]
            T_f[0] = S1_01_f * S0_10_f
            T_f[1] = S1_01_f * S0_11_f
            T_f[2] = S1_11_f * S0_10_f
            T_f[3] = S1_11_f * S0_11_f  ## dS1_01_f += dT_[0] * S0_10_f; dS0_10_f += dT_[0] * S1_01_f
            ## note that the S*_**_f are the only things that need to be stored in t he forward pass
            ## also note that there is an optimization here; should only need half

            T = ifft(T_f, output_array=T) ## dT_ = ifft(dT) (because DFT matrix symmetric)
            T *= subdiag[(n2 - 1)::(2 * n2), np.newaxis] ## dT *= subdiag[...]
            ## for learning A, should get something like dsubd[...] = T

            T_00, T_01, T_10, T_11 = T
            # polynomial additions
            T_00[:, n2:] += S_00[::2] ## dS_00[:n1,:] = T_00[:,n2:]
            T_00[:, n2:] += S_00[1::2]
            T_01[:, n2:] += S_01[::2]
            T_10[:, n2:] += S_10[1::2]

            ## autodiff correspondences annotated in with '##'
            ## this function takes in S and outputs T;
            ## the backwards pass calls these lines in reverse,
            ## taking dT and outputting dS where ## dx := \partial{L}/\partial{x},
            ## (L is the final output of the entire algorithm)
            T_prev = T

        return T[0].squeeze()[::-1]


class KrylovMultiply():
    """Multiply Krylov(A, v) @ w when A is zero except on the subdiagonal.
    """

    def __init__(self, n, lib='numpy'):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.fft_plans_forward = self.plan_ffts_forward_u_zero(m, lib)
        self.fft_plans_backward = self.plan_ffts_backward(m, lib)

    def plan_ffts_forward_u_zero(self, m, lib='numpy'):
        fft_plans = [None] * m
        for d in range(m)[::-1]:
            n1, n2 = 1<<d, 1<<(m-d)
            in_shape  = (3, n1, n2)
            out_shape = (2, n1, n2 // 2 + 1)
            fft_plans[d] = self._plan_ffts_forward_u_zero(in_shape, out_shape, lib)
        return fft_plans

    def _plan_ffts_forward_u_zero(self, in_shape, out_shape, lib='numpy'):
        if lib == 'numpy':
            S_time = np.zeros(shape=in_shape)
            fft_time2freq = lambda: np.fft.rfft(S_time)

            T_freq = np.empty(shape=out_shape, dtype='complex128')
            fft_freq2time = lambda: np.fft.irfft(T_freq)
            return ((S_time, fft_time2freq), (T_freq, fft_freq2time))
        if lib == 'scipy':
            pass
        if lib == 'fftw':
            in_shape_freq = in_shape[:-1] + (in_shape[-1]//2 + 1,)
            S_time = pyfftw.empty_aligned(in_shape, dtype='float64')
            S_freq = pyfftw.empty_aligned(in_shape_freq, dtype='complex128')
            fft_time2freq = pyfftw.FFTW(S_time, S_freq, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']) # don't destroy input so 0s are preserved
            S_time[:] = 0
            out_shape_freq = out_shape[:-1] + ((out_shape[-1] - 1) * 2,)
            T_time = pyfftw.empty_aligned(out_shape_freq, dtype='float64')
            T_freq = pyfftw.empty_aligned(out_shape, dtype='complex128')
            fft_freq2time = pyfftw.FFTW(T_freq, T_time, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
            return ((S_time, fft_time2freq), (T_freq, fft_freq2time))

    def plan_ffts_backward(self, m, lib='numpy'):
        fft_plans = [None] * m
        for d in range(m):
            n1, n2 = 1<<d, 1<<(m-d)
            in_shape  = (2, n1, n2)
            out_shape = (n1, n2 // 2 + 1)
            fft_plans[d] = self._plan_ffts_backward(in_shape, out_shape, lib)
        return fft_plans

    def _plan_ffts_backward(self, in_shape, out_shape, lib='numpy'):
        if lib == 'numpy':
            dT_time = np.empty(shape=in_shape)
            fft_time2freq = lambda: np.fft.ihfft(dT_time)

            dS_freq = np.zeros(shape=out_shape, dtype='complex128')
            fft_freq2time = lambda: np.fft.hfft(dS_freq)
            return ((dT_time, fft_time2freq), (dS_freq, fft_freq2time))
        if lib == 'scipy':
            pass
        if lib == 'fftw':
            in_shape_freq = in_shape[:-1] + (in_shape[-1]//2 + 1,)
            dT_time = pyfftw.empty_aligned(in_shape, dtype='float64')
            dT_freq = pyfftw.empty_aligned(in_shape_freq, dtype='complex128')
            # np.fft.ihfft is the same as np.fft.rfft().conj() / n
            fft_time2freq_func = pyfftw.FFTW(dT_time, dT_freq, direction='FFTW_FORWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])

            def fft_time2freq():
                dT_freq = fft_time2freq_func()
                np.conjugate(dT_freq, out=dT_freq)
                dT_freq /= in_shape[-1]
                return dT_freq

            out_shape_freq = out_shape[:-1] + ((out_shape[-1] - 1) * 2,)
            dS_time = pyfftw.empty_aligned(out_shape_freq, dtype='float64')
            dS_freq = pyfftw.empty_aligned(out_shape, dtype='complex128')
            # np.fft.hfft is the same as np.fft.irfft(input.conj()) * n
            fft_freq2time_func = pyfftw.FFTW(dS_freq, dS_time, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])

            def fft_freq2time():
                np.conjugate(dS_freq, out=dS_freq)
                return fft_freq2time_func(normalise_idft=False)

            return ((dT_time, fft_time2freq), (dS_freq, fft_freq2time))

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
            ((S_, fft_time2freq), (T_, fft_freq2time)) = self.fft_plans_forward[d]

            S0_10, S0_11, S1_11 = S_ ## pass
            S0_10[:,:n2] = S_10[::2]
            S0_11[:,:n2] = S_11[::2]
            S1_11[:,:n2] = S_11[1::2] ## dS_11[...] = dS1_11[...]

            # polynomial multiplications
            S0_10_f, S0_11_f, S1_11_f = fft_time2freq() ## dS_ = fft_time2freq(dS*_**_f)
            self.save_for_backward[d] = (S0_10_f, S0_11_f)

            T_[0] = S1_11_f * S0_10_f
            T_[1] = S1_11_f * S0_11_f  ## dS1_01_f += dT_[0] * S0_10_f; dS0_10_f += dT_[0] * S1_01_f
            ## note that the S*_**_f are the only things that need to be stored in t he forward pass
            ## also note that there is an optimization here; should only need half

            T = fft_freq2time() ## dT_ = fft_freq2time(dT) (because DFT matrix symmetric)
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
        self.save_for_backward = [None] * m
        self.forward(subdiag, v.reshape(n))
        v = v.reshape((n, 1))
        # We can ignore dT[2] and dT[3] because they start at zero and always stay zero.
        # We can check this by static analysis of the code or by math.
        (dT, _), _ = self.fft_plans_backward[0]
        dT[0], dT[1] = w[::-1].reshape((1, n)), np.zeros((1, n))
        # Temporarily allocate space for result, stored at self.fft_plans_backward[m]
        self.fft_plans_backward.append(((np.empty((2, n, 1)), None), (None, None)))

        for d in range(m):
            n1, n2 = 1 << d, 1 << (m - d - 1)
            (dT, fft_time2freq), (dS_f, fft_freq2time) = self.fft_plans_backward[d]
            (dS, _), _ = self.fft_plans_backward[d + 1]
            dT_00, dT_01 = dT
            dS_00, dS_01 = dS

            dS_00[::2] = dT_00[:, n2:]
            dS_00[1::2] = dT_00[:, n2:]
            dS_01[::2] = dT_01[:, n2:]
            dT *= subdiag[(n2 - 1)::(2 * n2), np.newaxis]

            dT_f = fft_time2freq()
            dS1_01_f = dS_f
            S0_10_f, S0_11_f = self.save_for_backward[d]
            dS1_01_f[:] = S0_10_f * dT_f[0]
            dS1_01_f += S0_11_f * dT_f[1]

            dS1_01 = fft_freq2time()
            dS_01[1::2] = dS1_01[:, :n2]

        self.fft_plans_backward = self.fft_plans_backward[:-1]
        du = (dS[0] * v + dS[1]).squeeze()
        return du
