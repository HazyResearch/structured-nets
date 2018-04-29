import functools

import torch
from torch.autograd import Variable

from cupy.cuda import cufft

from complex_utils import conjugate


@functools.lru_cache(maxsize=1024)
def cufft_plan1d(n, fft_type, batch_size):
    # Put in a separate function so we can cache the plans.
    # Actually this only saves about 2ms, from 65.5ms to 63.4ms, for n=4096, batchsize=512, and rank=3.
    return cufft.Plan1d(n, fft_type, batch_size)

def fft_raw(X):
    X = X.contiguous()
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    fft_type = cufft.CUFFT_C2C
    direction = cufft.CUFFT_FORWARD
    plan = cufft_plan1d(X.shape[-2], fft_type, X.numel() // X.shape[-2] // 2)
    out_shape = X.shape
    out = torch.cuda.FloatTensor(*out_shape)
    cufft.execC2C(plan.plan, X.data_ptr(), out.data_ptr(), direction)
    return out

def ifft_raw(X, norm=True):
    X = X.contiguous()
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    fft_type = cufft.CUFFT_C2C
    direction = cufft.CUFFT_INVERSE
    plan = cufft_plan1d(X.shape[-2], fft_type, X.numel() // X.shape[-2] // 2)
    out_shape = X.shape
    out = torch.cuda.FloatTensor(*out_shape)
    cufft.execC2C(plan.plan, X.data_ptr(), out.data_ptr(), direction)
    if norm:
        out /= X.shape[-2]
    return out

def rfft_raw(X):
    X = X.contiguous()
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    fft_type = cufft.CUFFT_R2C
    direction = cufft.CUFFT_FORWARD
    plan = cufft_plan1d(X.shape[-1], fft_type, X.numel() // X.shape[-1])
    out_shape = X.shape[:-1] + (X.shape[-1] // 2 + 1, 2)
    out = torch.cuda.FloatTensor(*out_shape)
    cufft.execR2C(plan.plan, X.data_ptr(), out.data_ptr())
    return out

def irfft_raw(X, norm=True):
    # TODO: Maybe support out_size because Rfft_slow backward might need it for odd size input.
    X = X.contiguous()
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    fft_type = cufft.CUFFT_C2R
    direction = cufft.CUFFT_INVERSE
    out_size = (X.shape[-2] - 1) * 2
    plan = cufft_plan1d(out_size, fft_type, X.numel() // X.shape[-2] // 2)
    out_shape = X.shape[:-2] + (out_size, )
    out = torch.cuda.FloatTensor(*out_shape)
    cufft.execC2R(plan.plan, X.data_ptr(), out.data_ptr())
    if norm:
        out /= out_size
    return out

class Fft(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return fft_raw(X)

    @staticmethod
    def backward(ctx, grad):
        return ifft_raw(grad, norm=False)

fft = Fft.apply


class Ifft(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return ifft_raw(X)

    @staticmethod
    def backward(ctx, grad):
        return fft_raw(grad) / grad.shape[-2]

ifft = Ifft.apply


class Rfft(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return rfft_raw(X)

    @staticmethod
    def backward(ctx, grad):
        input_size = (grad.shape[-2] - 1) * 2
        if input_size & 1:
            grad[..., 1:, :] /= 2
        elif input_size > 2:
            grad[..., 1:-1, :] /= 2
        return irfft(grad) * input_size

rfft = Rfft.apply

class Irfft(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return irfft_raw(X)

    @staticmethod
    def backward(ctx, grad):
        n = grad.shape[-1]
        grad_f = rfft_raw(grad) / n
        if grad_f.shape[-2] > 2:
            grad_f[..., 1:-1, :] *= 2
        return grad_f
        # return Ihfft.apply(grad)

irfft = Irfft.apply


def ihfft(X):
    # np.fft.ihfft is the same as np.fft.rfft().conj() / n
    return conjugate(rfft(X)) / X.shape[-1]

def hfft(X):
    # np.fft.hfft is the same as np.fft.irfft(input.conj()) * n
    return irfft(conjugate(X)) * (X.shape[-2] - 1) * 2

def torch_ihfft(X):
    # np.fft.ihfft is the same as np.fft.rfft().conj() / n
    return conjugate(torch.rfft(X, 1)) / X.shape[-1]

def torch_hfft(X):
    # np.fft.ihfft is the same as np.fft.rfft().conj() / n
    n = (X.shape[-2] - 1) * 2
    return torch.irfft(conjugate(X), 1, signal_sizes=(n, )) * (X.shape[-2] - 1) * 2
