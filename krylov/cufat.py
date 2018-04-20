import functools

import torch
from torch.autograd import Variable

import pytorch_fft.fft as pyfft
import cupy as cp
from cupy.cuda import cufft

CUPY_MEM = cp.ndarray(1, dtype='float32').data.mem


def torch_to_cupy(tensor):
    '''Super hacky way to convert torch.cuda.FloatTensor to CuPy array.
    Probably not safe, since we're manipulating GPU memory addresses directly.
    '''
    assert isinstance(tensor, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    # assert tensor.is_contiguous(), 'Input must be contiguous'
    offset = tensor.data_ptr() - CUPY_MEM.ptr
    array_mem = cp.cuda.memory.MemoryPointer(CUPY_MEM, offset)
    array = cp.ndarray(tensor.shape, dtype='float32', memptr=array_mem)
    array._strides = [4 * s for s in tensor.stride()]
    return array

def complex_multiply_cupy(X, Y):
    '''X and Y are complex64 tensors but stored as torch.cuda.FloatTensor.
    The even indices represent the real part and odd indices represent imaginary parts.
    '''
    assert isinstance(X, torch.cuda.FloatTensor) and isinstance(Y, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.shape[-1] % 2 == 0 and Y.shape[-1] % 2 == 0, 'Last dimension must be even'
    X_cp, Y_cp = torch_to_cupy(X), torch_to_cupy(Y)
    X_complex, Y_complex = X_cp.view('complex64'), Y_cp.view('complex64')
    out = torch.cuda.FloatTensor(*cp.broadcast(X_cp, Y_cp).shape)
    out_complex = torch_to_cupy(out).view('complex64')
    cp.multiply(X_complex, Y_complex, out_complex)
    return out

@functools.lru_cache(maxsize=1024)
def cufft_plan1d(n, fft_type, batch_size):
    # Put in a separate function so we can cache the plans.
    # Actually this only saves about 2ms, from 65.5ms to 63.4ms, for n=4096, batchsize=512, and rank=3.
    return cufft.Plan1d(n, fft_type, batch_size)

def fft(X):
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.is_contiguous(), 'Input must be contiguous'
    fft_type = cufft.CUFFT_C2C
    direction = cufft.CUFFT_FORWARD
    plan = cufft_plan1d(X.shape[-1] // 2, fft_type, X.numel() // X.shape[-1])
    out_shape = X.shape
    out = torch.cuda.FloatTensor(*out_shape)
    cufft.execC2C(plan.plan, X.data_ptr(), out.data_ptr(), direction)
    return out

def ifft(X):
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.is_contiguous(), 'Input must be contiguous'
    fft_type = cufft.CUFFT_C2C
    direction = cufft.CUFFT_INVERSE
    plan = cufft_plan1d(X.shape[-1] // 2, fft_type, X.numel() // X.shape[-1])
    out_shape = X.shape
    out = torch.cuda.FloatTensor(*out_shape)
    cufft.execC2C(plan.plan, X.data_ptr(), out.data_ptr(), direction)
    out /= X.shape[-1] // 2
    return out

def rfft(X):
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.is_contiguous(), 'Input must be contiguous'
    fft_type = cufft.CUFFT_R2C
    direction = cufft.CUFFT_FORWARD
    plan = cufft_plan1d(X.shape[-1], fft_type, X.numel() // X.shape[-1])
    out_shape = X.shape[:-1] + (2 * (X.shape[-1] // 2 + 1),)
    out = torch.cuda.FloatTensor(*out_shape)
    cufft.execR2C(plan.plan, X.data_ptr(), out.data_ptr())
    return out

def irfft(X, norm=True):
    # TODO: Maybe support out_size because Rfft backward might need it for odd size input.
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.is_contiguous(), 'Input must be contiguous'
    assert X.shape[-1] % 2 == 0, 'Last dimension must be even'
    fft_type = cufft.CUFFT_C2R
    direction = cufft.CUFFT_INVERSE
    out_size = (X.shape[-1] // 2 - 1) * 2
    plan = cufft_plan1d(out_size, fft_type, X.numel() // X.shape[-1])
    out_shape = X.shape[:-1] + (out_size,)
    out = torch.cuda.FloatTensor(*out_shape)
    cufft.execC2R(plan.plan, X.data_ptr(), out.data_ptr())
    if norm:
        out /= out_size
    return out

def ihfft(X):
    # np.fft.ihfft is the same as np.fft.rfft().conj() / n
    n = X.shape[-1]
    out = rfft(X)
    out_cp = torch_to_cupy(out).view('complex64')
    cp.conj(out_cp, out=out_cp)
    out /= n
    return out

def hfft(X):
    # np.fft.hfft is the same as np.fft.irfft(input.conj()) * n
    n = (X.shape[-1] // 2 - 1) * 2
    X_conj = torch.cuda.FloatTensor(*X.shape)
    X_cp, X_conj_cp = torch_to_cupy(X).view('complex64'), torch_to_cupy(X_conj).view('complex64')
    cp.conj(X_cp, out=X_conj_cp)
    return irfft(X_conj, norm=False)

def conjugate(X):
    # This doesn't work if X isn't contiguous
    assert X.shape[-1] % 2 == 0, 'Last dimension must be even'
    return (X.view(-1, 2) * torch.cuda.FloatTensor((1, -1))).view_as(X)

def conjugate_cupy(X):
    assert X.shape[-1] % 2 == 0, 'Last dimension must be even'
    X_cp = torch_to_cupy(X).view('complex64')
    X_conj = torch.cuda.FloatTensor(*X.shape)
    X_conj_cp = torch_to_cupy(X_conj).view('complex64')
    cp.conj(X_cp, out=X_conj_cp)
    return X_conj

def complex_mult_slow(X, Y):
    # I'm writing in this inefficient way because writing autodiff for complex mult is really hard
    # due to broadcasting. I'm just using Pytorch's functions here.
    real = X[..., ::2] * Y[..., ::2] - X[..., 1::2] * Y[..., 1::2]
    imag = X[..., ::2] * Y[..., 1::2] + X[..., 1::2] * Y[..., 0::2]
    result = torch.cat((real.view(-1, 1), imag.view(-1, 1)), dim=-1)
    return result.view(*(real.shape[:-1] + (2 * real.shape[-1], )))

class Conjugate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        # return conjugate(X)
        return conjugate_cupy(X)

    def backward(ctx, grad):
        return Conjugate.apply(grad)

# TODO: change this to be consistent: from now on e.g. Rfft should be fast and Rfft_slow should be slow
class Fft(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return fft(X.contiguous())

    @staticmethod
    def backward(ctx, grad):
        grad = grad.data.contiguous()
        return Fft.apply(grad)

class Ifft(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return ifft(X.contiguous())

    @staticmethod
    def backward(ctx, grad):
        grad = grad.data.contiguous()
        return Ifft.apply(grad)

class Rfft_fast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return rfft(X.contiguous())

    @staticmethod
    def backward(ctx, grad):
        input_size = (grad.shape[-1] // 2 - 1) * 2
        grad = grad.data.contiguous()
        if input_size & 1:
            grad[..., 2:] /= 2
        elif input_size > 2:
            grad[..., 2:-2] /= 2
        return Variable(irfft(grad) * input_size)
        # return Hfft_fast.apply(grad)

class Irfft_fast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return irfft(X.contiguous())

    @staticmethod
    def backward(ctx, grad):
        grad = grad.data.contiguous()
        n = grad.shape[-1]
        grad_f = rfft(grad) / n
        if grad_f.shape[-1] > 4:
            grad_f[..., 2:-2] *= 2
        return Variable(grad_f)
        # return Ihfft_fast.apply(grad)

def Ihfft_fast(X):
    # np.fft.ihfft is the same as np.fft.rfft().conj() / n
    n = X.shape[-1]
    return Conjugate.apply(Rfft_fast.apply(X)) / n

def Hfft_fast(X):
    # np.fft.hfft is the same as np.fft.irfft(input.conj()) * n
    n = (X.shape[-1] // 2 - 1) * 2
    return Irfft_fast.apply(Conjugate.apply(X)) * n


# class Ihfft_fast(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, X):
#         return ihfft(X.contiguous())

#     @staticmethod
#     def backward(ctx, grad):
#         return Irfft_fast.apply(grad)

# class Hfft_fast(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, X):
#         return hfft(X.contiguous())

#     @staticmethod
#     def backward(ctx, grad):
#         return Rfft_fast.apply(grad)

class ComplexMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return complex_multiply_cupy(X, Y)

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = complex_multiply_cupy(grad.data, conjugate_cupy(Y)), complex_multiply_cupy(grad.data, conjugate_cupy(X))
        # Need to sum over dimensions that were broadcasted
        dims_to_sum_X = [grad.dim() - i for i in range(1, X.dim() + 1) if X.shape[-i] != grad.shape[-i]]
        dims_to_sum_Y = [grad.dim() - i for i in range(1, Y.dim() + 1) if Y.shape[-i] != grad.shape[-i]]
        for dim in dims_to_sum_X:
            grad_X = grad_X.sum(dim=dim, keepdim=True)
        for dim in range(grad.dim() - X.dim())[::-1]:
            grad_X = grad_X.sum(dim=dim)
        for dim in dims_to_sum_Y:
            grad_Y = grad_Y.sum(dim=dim, keepdim=True)
        for dim in range(grad.dim() - Y.dim())[::-1]:
            grad_Y = grad_Y.sum(dim=dim)
        return Variable(grad_X), Variable(grad_Y)

class Rfft(torch.autograd.Function):
    def forward(self, X_re):
        X_re = X_re.contiguous()
        self._to_save_input_size = X_re.size(-1)
        return pyfft.rfft(X_re)

    def backward(self, grad_output_re, grad_output_im):
        # Clone the array and make contiguous if needed
        grad_output_re = pyfft.autograd.contiguous_clone(grad_output_re)
        grad_output_im = pyfft.autograd.contiguous_clone(grad_output_im)

        if self._to_save_input_size & 1:
            grad_output_re[...,1:] /= 2
        elif self._to_save_input_size > 2:
            grad_output_re[...,1:-1] /= 2

        if self._to_save_input_size & 1:
            grad_output_im[...,1:] /= 2
        elif self._to_save_input_size > 2:
            grad_output_im[...,1:-1] /= 2

        gr = pyfft.irfft(grad_output_re,grad_output_im,self._to_save_input_size, normalize=False)
        return gr

class Irfft(torch.autograd.Function):

    def forward(self, k_re, k_im):
        k_re, k_im = pyfft.autograd.make_contiguous(k_re, k_im)
        return pyfft.irfft(k_re, k_im)

    def backward(self, grad_output_re):
        grad_output_re = grad_output_re.contiguous()
        gr, gi = pyfft.rfft(grad_output_re)

        N = grad_output_re.size(-1)
        gr[...,0] /= N
        if gr.shape[-1] > 2:
            gr[...,1:-1] /= N/2
        gr[...,-1] /= N

        gi[...,0] /= N
        if gi.shape[-1] > 2:
            gi[...,1:-1] /= N/2
        gi[...,-1] /= N
        return gr, gi

def Ihfft(X_re):
    # np.fft.ihfft is the same as np.fft.rfft().conj() / n
    n = X_re.shape[-1]
    X_f_re, X_f_im = Rfft()(X_re)
    return X_f_re / n, -X_f_im / n

def Hfft(X_re, X_im):
    # np.fft.hfft is the same as np.fft.irfft(input.conj()) * n
    n = (X_re.shape[-1] - 1) * 2
    return Irfft()(X_re, -X_im) * n

def complex_mult(X, Y):
    X_re, X_im = X
    Y_re, Y_im = Y
    return X_re * Y_re - X_im * Y_im, X_re * Y_im + X_im * Y_re
