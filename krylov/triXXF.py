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

    def __init__(self, n, batch_size=1):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.plan_ffts_forward_pass()

    def plan_ffts_forward_pass(self):
        n, m, batch_size = self.n, self.m, self.batch_size
        self.S_storage = np.empty((m, 3 + batch_size, n))
        self.S_f_storage = [np.empty((3 + batch_size, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.T_f_storage = [np.empty((2 + 2 * batch_size, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.T_storage = np.empty((m, 2 + 2 * batch_size, n))
        self.ffts_forward_pass = []
        for d, (S, S_f, T_f, T) in enumerate(zip(self.S_storage, self.S_f_storage, self.T_f_storage, self.T_storage)):
            S = S.reshape((3 + batch_size, 1 << d, 1 << (m - d)))
            T = T.reshape((2 + 2 * batch_size, 1 << d, 1 << (m - d)))
            fft_time2freq = pyfftw.FFTW(S, S_f, direction='FFTW_FORWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'], threads=4)
            fft_freq2time = pyfftw.FFTW(T_f, T, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'], threads=4)
            self.ffts_forward_pass.append((fft_time2freq, fft_freq2time))

    def __call__(self, subdiag, v, u):
        """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
        We don't use bit reversal here.
        """
        n, m, batch_size = self.n, self.m, self.batch_size
        u, v = u.T.reshape(batch_size, n, 1), v.reshape(n, 1)
        self.S_storage.fill(0.0)
        T_prev = np.vstack((u * v, u, v[np.newaxis], np.ones((1, n, 1))))
        for d in range(m)[::-1]:
            n1, n2 = 1 << d, 1 << (m - d - 1)
            S = self.S_storage[d].reshape((3 + batch_size, n1, 2 * n2))
            S_f = self.S_f_storage[d]
            T_f = self.T_f_storage[d]
            T = self.T_storage[d].reshape((2 + 2 * batch_size, n1, 2 * n2))
            fft_time2freq, fft_freq2time = self.ffts_forward_pass[d]

            S_00, S_01, S_10, S_11 = T_prev[:batch_size], T_prev[batch_size:2 * batch_size], T_prev[-2], T_prev[-1]
            S0_10, S0_11, S1_01, S1_11 = S[0], S[1], S[2:2+ batch_size], S[-1]
            S0_10[:, :n2] = S_10[::2]
            S1_01[:, :, :n2] = S_01[:, 1::2]
            S0_11[:, :n2] = S_11[::2]
            S1_11[:, :n2] = S_11[1::2]

            # polynomial multiplications
            S_f = fft_time2freq(S, output_array=S_f) ## dS_ = fft(dS*_**_f)
            S0_10_f, S0_11_f, S1_01_f, S1_11_f = S_f[0], S_f[1], S_f[2:2+batch_size], S_f[-1]
            T_f[:batch_size] = S1_01_f * S0_10_f
            T_f[batch_size:2*batch_size] = S1_01_f * S0_11_f
            T_f[-2] = S1_11_f * S0_10_f
            T_f[-1] = S1_11_f * S0_11_f  ## dS1_01_f += dT_[0] * S0_10_f; dS0_10_f += dT_[0] * S1_01_f
            ## note that the S*_**_f are the only things that need to be stored in t he forward pass
            ## also note that there is an optimization here; should only need half

            T = fft_freq2time(T_f, output_array=T) ## dT_ = ifft(dT) (because DFT matrix symmetric)
            T *= subdiag[(n2 - 1)::(2 * n2), np.newaxis] ## dT *= subdiag[...]
            ## for learning A, should get something like dsubd[...] = T

            T_00, T_01, T_10, T_11 = T[:batch_size], T[batch_size:2 * batch_size], T[-2], T[-1]
            # polynomial additions
            T_00[:, :, n2:] += S_00[:, ::2] ## dS_00[:n1,:] = T_00[:,n2:]
            T_00[:, :, n2:] += S_00[:, 1::2]
            T_01[:, :, n2:] += S_01[:, ::2]
            T_10[:, n2:] += S_10[1::2]

            ## autodiff correspondences annotated in with '##'
            ## this function takes in S and outputs T;
            ## the backwards pass calls these lines in reverse,
            ## taking dT and outputting dS where ## dx := \partial{L}/\partial{x},
            ## (L is the final output of the entire algorithm)
            T_prev = T

        return T[:batch_size, :, ::-1].T.squeeze()


class KrylovMultiply():
    """Multiply Krylov(A, v) @ w when A is zero except on the subdiagonal.
    """

    def __init__(self, n, batch_size=1):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.plan_ffts_forward_pass_u_zero()
        self.plan_ffts_backward()

    def plan_ffts_forward_pass_u_zero(self):
        n, m = self.n, self.m
        self.S_storage = np.empty((m, 3, n))
        self.S_f_storage = [np.empty((3, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.T_f_storage = [np.empty((2, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.T_storage = np.empty((m, 2, n))
        self.ffts_forward_pass = []
        for d, (S, S_f, T_f, T) in enumerate(zip(self.S_storage, self.S_f_storage, self.T_f_storage, self.T_storage)):
            S = S.reshape((3, 1 << d, 1 << (m - d)))
            T = T.reshape((2, 1 << d, 1 << (m - d)))
            fft_time2freq = pyfftw.FFTW(S, S_f, direction='FFTW_FORWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
            fft_freq2time = pyfftw.FFTW(T_f, T, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
            self.ffts_forward_pass.append((fft_time2freq, fft_freq2time))

    def plan_ffts_backward(self):
        n, m, batch_size = self.n, self.m, self.batch_size
        self.dT_storage = np.empty((m + 1, 2 * batch_size, n))  # One extra array to store final result of backward pass
        self.dT_f_storage = [np.empty((2 * batch_size, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.dS_f_storage = [np.empty((batch_size, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.dS_storage = np.empty((m, batch_size, n))
        self.ffts_backward_pass = []
        for d, (dT, dT_f, dS_f, dS) in enumerate(zip(self.dT_storage, self.dT_f_storage, self.dS_f_storage, self.dS_storage)):
            dT = dT.reshape((2 * batch_size, 1 << d, 1 << (m - d)))
            dS = dS.reshape((batch_size, 1 << d, 1 << (m - d)))
            self.ffts_backward_pass.append((self.pyfftw_ihfft(dT, dT_f), self.pyfftw_hfft(dS_f, dS)))


    @staticmethod
    def pyfftw_ihfft(input_array, output_array):
        # np.fft.ihfft is the same as np.fft.rfft().conj() / n
        rfft = pyfftw.FFTW(input_array, output_array, direction='FFTW_FORWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
        def fft_time2freq(input_array=None, output_array=None):
            output_array = rfft(input_array, output_array)
            np.conjugate(output_array, out=output_array)
            output_array /= input_array.shape[-1]
            return output_array
        return fft_time2freq

    @staticmethod
    def pyfftw_hfft(input_array, output_array):
        # np.fft.hfft is the same as np.fft.irfft(input.conj()) * n
        irfft = pyfftw.FFTW(input_array, output_array, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
        def fft_freq2time(input_array=None, output_array=None):
            np.conjugate(input_array, out=input_array)
            return irfft(input_array, output_array, normalise_idft=False)
        return fft_freq2time

    def forward(self, subdiag, v):
        """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
        Special case when u = 0 to save intermediate results relating to v for
        the backward pass.
        """
        n, m = self.n, self.m
        self.save_for_backward = [None] * m
        v = v.reshape(n, 1)
        self.S_storage.fill(0.0)
        T_prev = (v, np.ones((n, 1)))
        for d in range(m)[::-1]:
            n1, n2 = 1 << d, 1 << (m - d - 1)
            S = self.S_storage[d].reshape((3, n1, 2 * n2))
            S_f = self.S_f_storage[d]
            T_f = self.T_f_storage[d]
            T = self.T_storage[d].reshape((2, n1, 2 * n2))
            fft_time2freq, fft_freq2time = self.ffts_forward_pass[d]

            S_10, S_11 = T_prev

            S0_10, S0_11, S1_11 = S
            S0_10[:,:n2] = S_10[::2]
            S0_11[:,:n2] = S_11[::2]
            S1_11[:,:n2] = S_11[1::2]

            # polynomial multiplications
            S0_10_f, S0_11_f, S1_11_f = fft_time2freq(S, output_array=S_f)
            self.save_for_backward[d] = (S0_10_f, S0_11_f)
            T_f[0] = S1_11_f * S0_10_f
            T_f[1] = S1_11_f * S0_11_f

            T = fft_freq2time(T_f, output_array=T)
            T *= subdiag[(n2 - 1)::(2 * n2), np.newaxis]

            T_10, T_11 = T
            # polynomial additions
            T_10[:,n2:] += S_10[1::2]

            T_prev = T

    def __call__(self, subdiag, v, w):
        n, m, batch_size = self.n, self.m, self.batch_size
        self.forward(subdiag, v.reshape(n))
        w, v = w.T.reshape(batch_size, 1, n), v.reshape((n, 1))
        # We can ignore dT[2] and dT[3] because they start at zero and always stay zero.
        # We can check this by static analysis of the code or by math.
        dT = self.dT_storage[0].reshape((2 * batch_size, 1, n))
        dT[:batch_size], dT[batch_size:] = w[:, :, ::-1], 0.0

        for d in range(m):
            n1, n2 = 1 << d, 1 << (m - d - 1)
            dT = self.dT_storage[d].reshape((2 * batch_size, n1, 2 * n2))
            dT_f = self.dT_f_storage[d]
            dS_f = self.dS_f_storage[d]
            dS = self.dS_storage[d].reshape((batch_size, n1, 2 * n2))
            fft_time2freq, fft_freq2time = self.ffts_backward_pass[d]
            dT_next = self.dT_storage[d + 1].reshape(2 * batch_size, 2 * n1, n2)

            dT_00, dT_01 = dT[:batch_size], dT[batch_size:]
            dS_00, dS_01 = dT_next[:batch_size], dT_next[batch_size:]

            dS_00[:, ::2] = dT_00[:, :, n2:]
            dS_00[:, 1::2] = dT_00[:, :, n2:]
            dS_01[:, ::2] = dT_01[:, :, n2:]
            dT *= subdiag[(n2 - 1)::(2 * n2), np.newaxis]

            dT_f = fft_time2freq(dT, output_array=dT_f)

            dS1_01_f  = dS_f
            S0_10_f, S0_11_f = self.save_for_backward[d]
            dS1_01_f[:] = S0_10_f * dT_f[:batch_size]
            dS1_01_f += S0_11_f * dT_f[batch_size:]

            dS1_01 = fft_freq2time(dS_f, output_array=dS)
            dS_01[:, 1::2] = dS1_01[:, :, :n2]
            # print(np.linalg.norm(dT_next[0] - dT_next[0, 0]))
            # dT[0] always contains the same polynomials. Maybe this is something we can exploit?
            # In the forward pass, this corresponds to the result only
            # depending on S_00[even] + S_00[odd], not their individual values.
            # Exploiting this will reduce the number of FFT calls from 8m to
            # 7m+1, but the FFT calls are no longer on the same dimension. It's
            # probably annoying to do and not worth it.

        du = (dT_next[:batch_size] * v + dT_next[batch_size:]).squeeze()
        # du = (w[0] * v + dT_next[1]).squeeze()
        return du.T
