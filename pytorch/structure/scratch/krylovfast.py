import numpy as np
import itertools

import pyfftw
import sys
sys.path.insert(0,'../../../pytorch/')
from structure.scratch.krylovslow import krylov_construct


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
    S0_10_mult_subdiag, S0_11, S1_01, S1_11 = S_ ## pass
    S0_10_mult_subdiag[:,:n2] = S_10[:n1,:]
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

    def __init__(self, n, batch_size=1, rank=1):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.rank = rank
        self.plan_ffts_forward_pass()

    def plan_ffts_forward_pass(self):
        n, m, batch_size, rank = self.n, self.m, self.batch_size, self.rank
        self.S_storage = [np.empty((batch_size + rank, n))] * m
        self.S_f_storage = [np.empty((batch_size + rank, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.T_f_storage = [np.empty((batch_size, rank, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.T_storage = [np.empty((batch_size, rank, 1 << (m - d))) for d in range(m)]
        self.ffts_forward_pass = []
        for d, (S, S_f, T_f, T) in enumerate(zip(self.S_storage, self.S_f_storage, self.T_f_storage, self.T_storage)):
            S = S.reshape((batch_size + rank, 1 << d, 1 << (m - d)))
            fft_time2freq = pyfftw.FFTW(S, S_f, direction='FFTW_FORWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'], threads=1)
            fft_freq2time = pyfftw.FFTW(T_f, T, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'], threads=1)
            self.ffts_forward_pass.append((fft_time2freq, fft_freq2time))

    def __call__(self, subdiag, v, u):
        """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
        We don't use bit reversal here.
        """
        n, m, batch_size, rank = self.n, self.m, self.batch_size, self.rank
        u, v = u.reshape(batch_size, n), v.reshape(rank, n)
        result = np.zeros((batch_size, rank, n), dtype=u.dtype)
        # T_00_sum = u @ v.T
        T_00_sum = (u[:, np.newaxis] * v).sum(axis=-1)
        result[:, :, 0] += T_00_sum
        T_01 = u.reshape(batch_size, n, 1).copy()  # Copy since we'll be changing this array directly
        T_10 = v.reshape(rank, n, 1)
        T_11 = np.ones(n)
        for d in range(m)[::-1]:
            n1, n2 = 1 << d, 1 << (m - d - 1)
            S = self.S_storage[d].reshape((batch_size + rank, n1, 2 * n2))
            S_f = self.S_f_storage[d]
            T_f = self.T_f_storage[d]
            T = self.T_storage[d]
            fft_time2freq, fft_freq2time = self.ffts_forward_pass[d]

            S_00_sum, S_01, S_10, S_11 = T_00_sum, T_01, T_10, T_11
            S[:, :, n2:] = 0.0
            S0_10_mult_subdiag, S1_01 = S[:rank, :, :n2], S[rank:rank + batch_size, :, :n2]
            S0_10_mult_subdiag[:] = S_10[:, ::2] * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
            S1_01[:] = S_01[:, 1::2]

            # polynomial multiplications
            S_f = fft_time2freq(S, output_array=S_f)
            S0_10_f, S1_01_f = S_f[:rank], S_f[rank:rank + batch_size]
            T_00_f_sum = T_f
            # T_00_f_sum[:] = (S1_01_f[:, np.newaxis] * S0_10_f[np.newaxis]).sum(axis=-2)
            np.einsum("bnm,rnm->brm", S1_01_f, S0_10_f, out=T_00_f_sum)
            T = fft_freq2time(T_f, output_array=T)
            T_00_sum = T

            # polynomial additions
            result[:, :, 1:2*n2] += T_00_sum[..., :-1]
            S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1)::(2 * n2)]
            # T_01 = np.concatenate((S_01[:, ::2], S_01[:, 1::2] * S0_11_mult_subdiag[:, np.newaxis]), axis=-1)
            T_01 = S_01.reshape(batch_size, n1, 2 * n2)
            T_01[:, :, n2:] *= S0_11_mult_subdiag[:, np.newaxis]
            T_10 = np.concatenate((S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, np.newaxis]), axis=-1)
            T_11 = S0_11_mult_subdiag * S_11[1::2]

        return result


class KrylovMultiply():
    """Multiply Krylov(A, v) @ w when A is zero except on the subdiagonal.
    """

    def __init__(self, n, batch_size=1, rank=1):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.rank = rank
        self.plan_ffts_forward_pass_u_zero()
        self.plan_ffts_backward_pass()

    def plan_ffts_forward_pass_u_zero(self):
        n, m, batch_size, rank = self.n, self.m, self.batch_size, self.rank
        self.S_storage = [np.empty((rank, n))] * m
        self.S_f_storage = [np.empty((rank, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.ffts_forward_pass = []
        for d, (S, S_f) in enumerate(zip(self.S_storage, self.S_f_storage)):
            S = S.reshape((rank, 1 << d, 1 << (m - d)))
            fft_time2freq = pyfftw.FFTW(S, S_f, direction='FFTW_FORWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'], threads=1)
            self.ffts_forward_pass.append(fft_time2freq)

    def plan_ffts_backward_pass(self):
        n, m, batch_size, rank = self.n, self.m, self.batch_size, self.rank
        self.dT_storage = [np.empty((batch_size, rank, 1 << (m - d))) for d in range(m)]
        self.dT_f_storage = [np.empty((batch_size, rank, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.dS_f_storage = [np.empty((batch_size, 1 << d, (1 << (m - d - 1)) + 1), dtype='complex128') for d in range(m)]
        self.dS_storage = [np.empty((batch_size, n))] * m
        self.ffts_backward_pass = []
        for d, (dT, dT_f, dS_f, dS) in enumerate(zip(self.dT_storage, self.dT_f_storage, self.dS_f_storage, self.dS_storage)):
            dS = dS.reshape((batch_size, 1 << d, 1 << (m - d)))
            fft_time2freq = pyfftw.FFTW(dT, dT_f, direction='FFTW_FORWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'], threads=1)
            fft_freq2time = pyfftw.FFTW(dS_f, dS, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'], threads=1)
            self.ffts_backward_pass.append((fft_time2freq, fft_freq2time))

    def __call__(self, subdiag, v, w):
        n, m, batch_size, rank = self.n, self.m, self.batch_size, self.rank
        # Forward pass. Since K @ w can be computed by autodiffing K^T @ u, we
        # carry out the forward pass K^T @ u for u = 0 here to save the
        # intermediate values. This code is exactly the same as the function
        # @krylov_transpose_multiply, specialized to the case where u = 0.
        save_for_backward = [None] * m
        T_10 = v.reshape(rank, n, 1)
        T_11 = np.ones(n)
        for d in range(m)[::-1]:
            n1, n2 = 1 << d, 1 << (m - d - 1)
            S = self.S_storage[d].reshape((rank, n1, 2 * n2))
            S_f = self.S_f_storage[d]
            fft_time2freq = self.ffts_forward_pass[d]
            S_10, S_11 = T_10, T_11
            S0_10_mult_subdiag = S[:, :, :n2]
            S0_10_mult_subdiag[:] = S_10[:, ::2] * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
            S[:, :, n2:] = 0.0
            S0_10_mult_subdiag_f = fft_time2freq(S, output_array=S_f)
            T_10 = np.concatenate((S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, np.newaxis]), axis=-1)
            S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1)::(2 * n2)]
            save_for_backward[d] = S0_10_mult_subdiag_f, S0_11_mult_subdiag
            T_11 = S0_11_mult_subdiag * S_11[1::2]

        # Backward pass
        w, v = w.reshape(batch_size, rank, n), v.reshape((rank, n))
        dT_01 = np.zeros((batch_size, 1, n), dtype=w.dtype)

        for d in range(m):
            n1, n2 = 1 << d, 1 << (m - d - 1)
            dT = self.dT_storage[d]
            dT_f = self.dT_f_storage[d]
            dS_f = self.dS_f_storage[d]
            dS = self.dS_storage[d].reshape((batch_size, n1, 2 * n2))
            fft_time2freq, fft_freq2time = self.ffts_backward_pass[d]

            S0_10_mult_subdiag_f, S0_11_mult_subdiag = save_for_backward[d]
            dS_01 = np.empty((batch_size, 2 * n1, n2), dtype=w.dtype)
            dS_01[:, ::2] = dT_01[:, :, :n2]
            dT_00_sum = dT
            dT_00_sum[:, :, :2*n2 - 1] = w[:, :, 1:2*n2]
            dT_00_sum[:, :, -1] = 0.0

            dT_00_sum_f = fft_time2freq(dT, output_array=dT_f)
            dS1_01_f = dS_f
            # dS1_01_f[:] = (np.conjugate(S0_10_mult_subdiag_f, out=S0_10_mult_subdiag_f) * dT_00_sum_f[:, :, np.newaxis]).sum(axis=1)
            np.einsum("brm,rnm->bnm", dT_00_sum_f, np.conjugate(S0_10_mult_subdiag_f, out=S0_10_mult_subdiag_f), out=dS1_01_f)

            dS1_01 = fft_freq2time(dS_f, output_array=dS)
            dS_01[:, 1::2] = dT_01[:, :, n2:] * S0_11_mult_subdiag[:, np.newaxis] + dS1_01[:, :, :n2]
            dT_01 = dS_01

        # du = ((dT_00_sum[:, :, np.newaxis] * v[np.newaxis, :, :, np.newaxis]).sum(dim=1) + dT_01).squeeze(axis=-1)
        du = w[:, :, 0] @ v + dT_01.squeeze(axis=-1)
        return du


def test_krylov_transpose_multiply():
    m = 14
    n = 1<<m
    batch_size = 3
    rank = 2
    subdiag = np.random.random(n-1)
    A = np.diag(subdiag, -1)
    u = np.random.random((batch_size, n))
    v = np.random.random((rank, n))
    # k1 = krylov_mult_slow(A,v,u,n)
    # k1_allocated = krylov_mult_slow_allocated(A,v,u,n)
    # k11 = krylov_mult_slow_faster(A,v,u,n)
    # k2 = krylov_mult(A,v,u,n)
    resolvent_bilinear_flattened = create(n, m, lib='fftw')
    krylov_transpose_multiply = KrylovTransposeMultiply(n, batch_size, rank)
    k3 = resolvent_bilinear_flattened(A, v[0], u[0], n, m)
    k3_nobf = krylov_transpose_multiply(subdiag, v, u)
    assert np.allclose(k3, k3_nobf[0, 0])


def test_krylov_multiply():
    m = 14
    n = 1<<m
    batch_size = 3
    rank = 2
    subdiag = np.random.random(n-1)
    A = np.diag(subdiag, -1)
    w = np.random.random((batch_size, rank, n))
    v = np.random.random((rank, n))
    krylov_multiply = KrylovMultiply(n, batch_size, rank)
    result1 = krylov_multiply(subdiag, v, w)
    Ks = [krylov_construct(A, v[i], n) for i in range(rank)]
    result2 = np.stack([w[:, i] @ Ks[i] for i in range(rank)]).swapaxes(0, 1).sum(axis=1)
    assert np.allclose(result1, result2)


def main():
    test_krylov_transpose_multiply()
    test_krylov_multiply()


if __name__ == '__main__':
    main()
