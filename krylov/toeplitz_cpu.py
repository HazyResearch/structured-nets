import numpy as np
import scipy.fftpack as fft
import itertools
from scipy import signal


class KT_Toeplitz():
    """Multiply Krylov(A, v)^T @ u when A is zero except on the subdiagonal.
    """

    def __init__(self, n, f=0, batch_size=1, rank=1):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.rank = rank

        self.eta = None
        if f == 0:
            pass
        else:
            mod = np.power(np.abs(f), np.arange(n)/n)
            if f > 0:
                arg = np.ones(n)
            else:
                arg = np.fft.fft(np.eye(1,2*n,2*n-1))[0,:n]
            self.eta = mod * arg


    def mult(self, v, u): # assume rank 1, batch 1 for now
        u_ = np.fft.ifft(1/self.eta * u)
        v_ = np.fft.fft(self.eta * v)
        ans = self.eta * np.fft.fft(w_ * v_)
        return np.real(ans)

    def __call__(self, v, u):
        """
        Multiply Krylov(Z_f, v)^T @ u
        v: (rank, n)
        u: (batch, n)
        """
        n, m, batch_size, rank = self.n, self.m, self.batch_size, self.rank
        u_ = np.fft.ifft(1/self.eta * u)
        v_ = np.fft.fft(self.eta * v)
        uv_ = u_.reshape(batch_size, 1, n) * v_.reshape(1, rank, n)
        ans = self.eta * np.fft.fft(uv_)
        return np.real(ans)


class K_Toeplitz():
    """Multiply Krylov(A, v) @ w when A is zero except on the subdiagonal.
    """

    def __init__(self, n, f, batch_size=1, rank=1):
        m = int(np.log2(n))
        assert n == 1 << m, 'n must be a power of 2'
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.rank = rank

        self.eta = None
        if f == 0:
            pass
        else:
            mod = np.power(np.abs(f), np.arange(n)/n)
            if f > 0:
                arg = np.ones(n)
            else:
                arg = np.fft.fft(np.eye(1,2*n,2*n-1))[0,:n]
            self.eta = mod * arg

    def mult(self, v, w): # assume rank 1, batch 1 for now
        w_ = np.fft.fft(self.eta * w)
        v_ = np.fft.fft(self.eta * v)
        ans = 1/self.eta * np.fft.ifft(w_ * v_)
        return np.real(ans)

    def __call__(self, v, w):
        """
        v: (rank, n)
        w: (batch_size, rank, n)
        """
        n, m, batch_size, rank = self.n, self.m, self.batch_size, self.rank
        w_ = np.fft.fft(self.eta * w)
        v_ = np.fft.fft(self.eta * v)
        wv_ = v_.reshape((1, rank, n)) * w_
        ans = 1/self.eta * np.fft.ifft(np.sum(wv_, axis=1))
        return np.real(ans)
