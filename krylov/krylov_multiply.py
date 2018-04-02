import functools
import numpy as np

import torch
from torch.autograd import Variable
import pytorch_fft.fft as fft

import cupy as cp
from cupy.cuda import cufft

CUPY_MEM = cp.ndarray((1, ), dtype='float32').data.mem

from triextrafat import krylov_construct
# from triXXF import KrylovTransposeMultiply

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

def rfft(X):
    assert isinstance(X, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.is_contiguous(), 'Input must be contiguous'
    fft_type = cufft.CUFFT_R2C
    direction = cufft.CUFFT_FORWARD
    plan = cufft.Plan1d(X.shape[-1], fft_type, X.numel() // X.shape[-1])
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
    plan = cufft.Plan1d(out_size, fft_type, X.numel() // X.shape[-1])
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
    return (X.view(-1, 2) * torch.cuda.FloatTensor((1, -1))).view_as(X)

def conjugate_cupy(X):
    X_cp = torch_to_cupy(X).view('complex64')
    X_conj = torch.cuda.FloatTensor(*X.shape)
    X_conj_cp = torch_to_cupy(X_conj).view('complex64')
    cp.conj(X_cp, out=X_conj_cp)
    return X_conj

def complex_mult_slow(X, Y):
    # I'm writing in this efficient many because writing autodiff for complex mult is really hard
    # thanks for broadcasting. I'm just using Pytorch's functions here.
    real = X[..., ::2] * Y[..., ::2] - X[..., 1::2] * Y[..., 1::2]
    imag = X[..., ::2] * Y[..., 1::2] + X[..., 1::2] * Y[..., 0::2]
    result = torch.cat((real.view(-1, 1), imag.view(-1, 1)), dim=-1)
    return result.view(*(real.shape[:-1] + (2 * real.shape[-1], )))

class Conjugate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return conjugate(X)
        # return conjugate_cupy(X)

    def backward(ctx, grad):
        return Conjugate.apply(grad)

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
        # TODO: This is not correct if the multiplication was broadcasted
        X, Y = ctx.saved_tensors
        # There's probably a better way to do this, maybe define a Conjugate function
        X_conj = torch.cuda.FloatTensor(*X.shape)
        X_cp, X_conj_cp = torch_to_cupy(X).view('complex64'), torch_to_cupy(X_conj).view('complex64')
        cp.conj(X_cp, out=X_conj_cp)
        Y_conj = torch.cuda.FloatTensor(*Y.shape)
        Y_cp, Y_conj_cp = torch_to_cupy(Y).view('complex64'), torch_to_cupy(Y_conj).view('complex64')
        cp.conj(Y_cp, out=Y_conj_cp)
        return Variable(complex_multiply_cupy(grad.data, Y_conj)), Variable(complex_multiply_cupy(grad.data, X_conj))

class Rfft(torch.autograd.Function):
    def forward(self, X_re):
        X_re = X_re.contiguous()
        self._to_save_input_size = X_re.size(-1)
        return fft.rfft(X_re)

    def backward(self, grad_output_re, grad_output_im):
        # Clone the array and make contiguous if needed
        grad_output_re = fft.autograd.contiguous_clone(grad_output_re)
        grad_output_im = fft.autograd.contiguous_clone(grad_output_im)

        if self._to_save_input_size & 1:
            grad_output_re[...,1:] /= 2
        elif self._to_save_input_size > 2:
            grad_output_re[...,1:-1] /= 2

        if self._to_save_input_size & 1:
            grad_output_im[...,1:] /= 2
        elif self._to_save_input_size > 2:
            grad_output_im[...,1:-1] /= 2

        gr = fft.irfft(grad_output_re,grad_output_im,self._to_save_input_size, normalize=False)
        return gr

class Irfft(torch.autograd.Function):

    def forward(self, k_re, k_im):
        k_re, k_im = fft.autograd.make_contiguous(k_re, k_im)
        return fft.irfft(k_re, k_im)

    def backward(self, grad_output_re):
        grad_output_re = grad_output_re.contiguous()
        gr, gi = fft.rfft(grad_output_re)

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

def krylov_transpose_multiply(subdiag, v, u):
    """Multiply Krylov(A, v_i)^T @ u when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    T_00 = u[:, np.newaxis, ..., np.newaxis] * v[np.newaxis, ..., np.newaxis]
    T_01 = u[..., np.newaxis]
    T_10 = v[..., np.newaxis]
    # T_11 = Variable(torch.ones((n, 1))).cuda()
    T_11 = Variable(torch.cuda.FloatTensor(n, 1).fill_(1.0))
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_00, S_01, S_10, S_11 = T_00, T_01, T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S1_01 = torch.cat((S_01[:, 1::2], torch.zeros_like(S_01[:, 1::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_01, S1_11[np.newaxis]))

        # polynomial multiplications
        S_f_re, S_f_im = Rfft()(S)
        S0_10_f_re, S0_11_f_re, S1_01_f_re, S1_11_f_re = S_f_re[:rank], S_f_re[rank], S_f_re[rank+1:rank+1+batch_size], S_f_re[-1]
        S0_10_f_im, S0_11_f_im, S1_01_f_im, S1_11_f_im = S_f_im[:rank], S_f_im[rank], S_f_im[rank+1:rank+1+batch_size], S_f_im[-1]
        S1_01_f = (S1_01_f_re, S1_01_f_im)
        S0_11_f = (S0_11_f_re, S0_11_f_im)
        S1_11_f = (S1_11_f_re, S1_11_f_im)
        S0_10_f = (S0_10_f_re, S0_10_f_im)
        T_00_f_re, T_00_f_im = complex_mult((S1_01_f_re[:, np.newaxis], S1_01_f_im[:, np.newaxis]),
                                            (S0_10_f_re[np.newaxis], S0_10_f_im[np.newaxis]))
        T_01_f_re, T_01_f_im = complex_mult(S1_01_f, S0_11_f)
        T_10_f_re, T_10_f_im = complex_mult(S1_11_f, S0_10_f)
        T_11_f_re, T_11_f_im = complex_mult(S1_11_f, S0_11_f)

        T_f_re = torch.cat((torch.cat((T_00_f_re, T_01_f_re[:, np.newaxis]), dim=1),
                            torch.cat((T_10_f_re[np.newaxis], T_11_f_re[np.newaxis, np.newaxis]), dim=1)))
        T_f_im = torch.cat((torch.cat((T_00_f_im, T_01_f_im[:, np.newaxis]), dim=1),
                            torch.cat((T_10_f_im[np.newaxis], T_11_f_im[np.newaxis, np.newaxis]), dim=1)))

        T = Irfft()(T_f_re, T_f_im) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_00, T_01, T_10, T_11 = T[:batch_size, :rank], T[:batch_size, -1], T[-1, :rank], T[-1, -1]

        # polynomial additions
        T_00 = torch.cat((T_00[:, :, :, :n2], T_00[:, :, :, n2:] + S_00[:, :, ::2] + S_00[:, :, 1::2]), dim=-1)
        T_01 = torch.cat((T_01[:, :, :n2], T_01[:, :, n2:] + S_01[:, ::2]), dim=-1)
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    # Negative step isn't supported by Pytorch
    # (https://github.com/pytorch/pytorch/issues/229) so we have to construct
    # the index explicitly.
    # reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    reverse_index = torch.arange(n - 1, -1, -1, out=torch.cuda.LongTensor())
    return T_00[:, :, :, reverse_index].squeeze(dim=2)


def krylov_transpose_multiply_fast(subdiag, v, u):
    """Multiply Krylov(A, v_i)^T @ u when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    T_00 = u[:, np.newaxis, ..., np.newaxis] * v[np.newaxis, ..., np.newaxis]
    T_01 = u[..., np.newaxis]
    T_10 = v[..., np.newaxis]
    # T_11 = Variable(torch.ones((n, 1))).cuda()
    T_11 = Variable(torch.cuda.FloatTensor(n, 1).fill_(1.0))
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_00, S_01, S_10, S_11 = T_00, T_01, T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S1_01 = torch.cat((S_01[:, 1::2], torch.zeros_like(S_01[:, 1::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_01, S1_11[np.newaxis]))

        # polynomial multiplications
        S_f = Rfft_fast.apply(S)
        S0_10_f, S0_11_f, S1_01_f, S1_11_f = S_f[:rank], S_f[rank], S_f[rank+1:rank+1+batch_size], S_f[-1]
        # T_00_f = ComplexMult.apply(S1_01_f[:, np.newaxis], S0_10_f[np.newaxis])
        # T_01_f = ComplexMult.apply(S1_01_f, S0_11_f)
        # T_10_f = ComplexMult.apply(S1_11_f, S0_10_f)
        # T_11_f = ComplexMult.apply(S1_11_f, S0_11_f)
        T_00_f = complex_mult_slow(S1_01_f[:, np.newaxis], S0_10_f[np.newaxis])
        T_01_f = complex_mult_slow(S1_01_f, S0_11_f)
        T_10_f = complex_mult_slow(S1_11_f, S0_10_f)
        T_11_f = complex_mult_slow(S1_11_f, S0_11_f)

        T_f = torch.cat((torch.cat((T_00_f, T_01_f[:, np.newaxis]), dim=1),
                         torch.cat((T_10_f[np.newaxis], T_11_f[np.newaxis, np.newaxis]), dim=1)))

        T = Irfft_fast.apply(T_f) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_00, T_01, T_10, T_11 = T[:batch_size, :rank], T[:batch_size, -1], T[-1, :rank], T[-1, -1]

        # polynomial additions
        T_00 = torch.cat((T_00[:, :, :, :n2], T_00[:, :, :, n2:] + S_00[:, :, ::2] + S_00[:, :, 1::2]), dim=-1)
        T_01 = torch.cat((T_01[:, :, :n2], T_01[:, :, n2:] + S_01[:, ::2]), dim=-1)
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    # Negative step isn't supported by Pytorch
    # (https://github.com/pytorch/pytorch/issues/229) so we have to construct
    # the index explicitly.
    # reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    reverse_index = torch.arange(n - 1, -1, -1, out=torch.cuda.LongTensor())
    return T_00[:, :, :, reverse_index].squeeze(dim=2)


def krylov_multiply_by_autodiff(subdiag, v, w):
    """Multiply \sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal, using Pytorch's autodiff.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    # u = Variable(torch.zeros((batch_size, n)).cuda(), requires_grad=True)
    u = Variable(torch.cuda.FloatTensor(batch_size, n).fill_(0.0), requires_grad=True)
    prod = krylov_transpose_multiply_fast(subdiag, v, u)
    result, = torch.autograd.grad(prod, u, grad_outputs=w, retain_graph=True)
    return result

def krylov_multiply_forward(subdiag, v):
    rank, n = v.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = [None] * m
    T_10 = v[..., np.newaxis]
    # T_11 = Variable(torch.ones((n, 1))).cuda()
    T_11 = Variable(torch.cuda.FloatTensor(n, 1).fill_(1.0))
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_11[np.newaxis]))

        # polynomial multiplications
        S_f_re, S_f_im = Rfft()(S)
        S0_10_f_re, S0_11_f_re, S1_11_f_re = S_f_re[:rank], S_f_re[-2], S_f_re[-1]
        S0_10_f_im, S0_11_f_im, S1_11_f_im = S_f_im[:rank], S_f_im[-2], S_f_im[-1]
        S0_10_f = (S0_10_f_re, S0_10_f_im)
        S0_11_f = (S0_11_f_re, S0_11_f_im)
        S1_11_f = (S1_11_f_re, S1_11_f_im)
        save_for_backward[d] = (S0_10_f, S0_11_f)

        T_10_f_re, T_10_f_im = complex_mult(S1_11_f, S0_10_f)
        T_11_f_re, T_11_f_im = complex_mult(S1_11_f, S0_11_f)

        T_f_re = torch.cat((T_10_f_re, T_11_f_re[np.newaxis]))
        T_f_im = torch.cat((T_10_f_im, T_11_f_im[np.newaxis]))

        T = Irfft()(T_f_re, T_f_im) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_10, T_11 = T[:rank], T[-1]

        # polynomial additions
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    return save_for_backward

def krylov_multiply_forward_fast(subdiag, v):
    rank, n = v.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = [None] * m
    T_10 = v[..., np.newaxis]
    # T_11 = Variable(torch.ones((n, 1))).cuda()
    T_11 = Variable(torch.cuda.FloatTensor(n, 1).fill_(1.0))
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10 = torch.cat((S_10[:, ::2], torch.zeros_like(S_10[:, ::2])), dim=-1)
        S0_11 = torch.cat((S_11[::2], torch.zeros_like(S_11[::2])), dim=-1)
        S1_11 = torch.cat((S_11[1::2], torch.zeros_like(S_11[1::2])), dim=-1)
        S = torch.cat((S0_10, S0_11[np.newaxis], S1_11[np.newaxis]))

        # polynomial multiplications
        S_f = Rfft_fast.apply(S)
        S0_10_f, S0_11_f, S1_11_f = S_f[:rank], S_f[-2], S_f[-1]
        save_for_backward[d] = (S0_10_f, S0_11_f)

        # T_10_f = ComplexMult.apply(S1_11_f, S0_10_f)
        # T_11_f = ComplexMult.apply(S1_11_f, S0_11_f)
        T_10_f = complex_mult_slow(S1_11_f, S0_10_f)
        T_11_f = complex_mult_slow(S1_11_f, S0_11_f)

        T_f = torch.cat((T_10_f, T_11_f[np.newaxis]))

        T = Irfft_fast.apply(T_f) * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_10, T_11 = T[:rank], T[-1]

        # polynomial additions
        T_10 = torch.cat((T_10[:, :, :n2], T_10[:, :, n2:] + S_10[:, 1::2]), dim=-1)

    return save_for_backward

def krylov_multiply(subdiag, v, w):
    """Multiply \sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = krylov_multiply_forward(subdiag, v)
    # reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    reverse_index = torch.arange(n - 1, -1, -1, out=torch.cuda.LongTensor())
    w = w.view(batch_size, rank, 1, n)
    # dT_00, dT_01 = w[:, :, :, reverse_index], Variable(torch.zeros((batch_size, 1, n)).cuda())
    dT_00, dT_01 = w[:, :, :, reverse_index], Variable(torch.cuda.FloatTensor(batch_size, 1, n).fill_(0.0))

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        dS_00 = Variable(torch.cuda.FloatTensor(batch_size, rank, 2 * n1, n2))
        dS_00[:, :, ::2] = dT_00[:, :, :, n2:]
        dS_00[:, :, 1::2] = dT_00[:, :, :, n2:]
        dS_01 = Variable(torch.cuda.FloatTensor(batch_size, 2 * n1, n2))
        dS_01[:, ::2] = dT_01[:, :, n2:]

        dT = torch.cat((dT_00, dT_01[:, np.newaxis]), dim=1)
        dT = dT * subdiag[(n2 - 1)::(2 * n2), np.newaxis]

        dT_f_re, dT_f_im = Ihfft(dT)
        dT_00_f_re, dT_01_f_re = dT_f_re[:, :rank], dT_f_re[:, -1]
        dT_00_f_im, dT_01_f_im = dT_f_im[:, :rank], dT_f_im[:, -1]
        dT_00_f = (dT_00_f_re, dT_00_f_im)
        dT_01_f = (dT_01_f_re, dT_01_f_im)

        S0_10_f, S0_11_f = save_for_backward[d]
        S0_10_f_re, S0_10_f_im = S0_10_f
        dS1_01_f_re, dS1_01_f_im = complex_mult((S0_10_f_re[np.newaxis], S0_10_f_im[np.newaxis]),
                                                 dT_00_f)
        prod_re, prod_im = complex_mult(S0_11_f, dT_01_f)
        dS1_01_f_re = dS1_01_f_re.sum(dim=1) + prod_re
        dS1_01_f_im = dS1_01_f_im.sum(dim=1) + prod_im

        dS1_01 = Hfft(dS1_01_f_re, dS1_01_f_im)
        dS_01[:, 1::2] = dS1_01[:, :, :n2]

        dT_00, dT_01 = dS_00, dS_01

    du = ((dT_00 * v[np.newaxis, :, :, np.newaxis]).sum(dim=1) + dT_01).squeeze(dim=-1)
    return du


def krylov_multiply_fast(subdiag, v, w):
    """Multiply \sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        w: Tensor of shape (batch_size, rank, n)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = krylov_multiply_forward_fast(subdiag, v)
    # reverse_index = torch.arange(n - 1, -1, -1).long().cuda()
    reverse_index = torch.arange(n - 1, -1, -1, out=torch.cuda.LongTensor())
    w = w.view(batch_size, rank, 1, n)
    # dT_00, dT_01 = w[:, :, :, reverse_index], Variable(torch.zeros((batch_size, 1, n)).cuda())
    dT_00, dT_01 = w[:, :, :, reverse_index], Variable(torch.cuda.FloatTensor(batch_size, 1, n).fill_(0.0))

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        dS_00 = Variable(torch.cuda.FloatTensor(batch_size, rank, 2 * n1, n2))
        dS_00[:, :, ::2] = dT_00[:, :, :, n2:]
        dS_00[:, :, 1::2] = dT_00[:, :, :, n2:]
        dS_01 = Variable(torch.cuda.FloatTensor(batch_size, 2 * n1, n2))
        dS_01[:, ::2] = dT_01[:, :, n2:]

        dT = torch.cat((dT_00, dT_01[:, np.newaxis]), dim=1)
        dT = dT * subdiag[(n2 - 1)::(2 * n2), np.newaxis]

        # dT_f = Ihfft_fast.apply(dT)
        dT_f = Ihfft_fast(dT)
        dT_00_f, dT_01_f = dT_f[:, :rank], dT_f[:, -1]

        S0_10_f, S0_11_f = save_for_backward[d]
        # dS1_01_f = ComplexMult.apply(S0_10_f[np.newaxis], dT_00_f).sum(dim=1) + ComplexMult.apply(S0_11_f, dT_01_f)
        dS1_01_f = complex_mult_slow(S0_10_f[np.newaxis], dT_00_f).sum(dim=1) + complex_mult_slow(S0_11_f, dT_01_f)

        # dS1_01 = Hfft_fast.apply(dS1_01_f)
        dS1_01 = Hfft_fast(dS1_01_f)
        dS_01[:, 1::2] = dS1_01[:, :, :n2]

        dT_00, dT_01 = dS_00, dS_01

    du = ((dT_00 * v[np.newaxis, :, :, np.newaxis]).sum(dim=1) + dT_01).squeeze(dim=-1)
    return du


def test_tranpose_multiply():
    m = 12
    n = 1<<m
    batch_size = 512
    rank = 3
    subdiag = Variable(torch.rand(n-1), requires_grad=True).cuda()
    A = np.diag(subdiag.data.cpu().numpy(), -1)
    u = Variable(torch.rand((batch_size, n)), requires_grad=True).cuda()
    v = Variable(torch.rand((rank, n)), requires_grad=True).cuda()
    result = krylov_transpose_multiply_fast(subdiag, v, u)
    grad,  = torch.autograd.grad(torch.sum(result), v, retain_graph=True)
    grad = grad.data.cpu().numpy()
    result = result.data.cpu().numpy()
    result1 = krylov_transpose_multiply(subdiag, v, u)
    grad1, = torch.autograd.grad(torch.sum(result1), v, retain_graph=True)
    grad1 = grad1.data.cpu().numpy()
    result1 = result1.data.cpu().numpy()
    # CPU dense multiply
    Ks = [krylov_construct(A, v.data.cpu().numpy()[i], n) for i in range(rank)]
    u_cpu = u.data.cpu().numpy()
    result2 = np.stack([u_cpu @ K.T for K in Ks])
    result2 = result2.swapaxes(0, 1).squeeze()
    # GPU dense multiply
    Ks_pytorch = [Variable(torch.Tensor(K)).cuda() for K in Ks]
    result3 = torch.stack([u @ K.t() for K in Ks_pytorch])
    result3 = result3.data.cpu().numpy().swapaxes(0, 1).squeeze()
    # Explicit construction on GPU
    linear_fn = functools.partial(shift_subdiag, subdiag)
    Ks_gpu = [Krylov(linear_fn, v_) for v_ in v]
    result4 = torch.stack([u @ K for K in Ks_gpu])
    result4 = result4.data.cpu().numpy().swapaxes(0, 1).squeeze()
    # np.allclose(result1, result2)
    print(np.max(abs(result - result1)))
    print(np.mean(abs(result - result1)))
    print(np.max(abs(grad - grad1)))
    print(np.mean(abs(grad - grad1)))
    print(np.max(abs(result1 - result2)))
    print(np.mean(abs(result1 - result2)))
    print(np.max(abs(result3 - result2)))
    print(np.mean(abs(result3 - result2)))
    print(np.max(abs(result4 - result2)))
    print(np.mean(abs(result4 - result2)))

def test_multiply():
    m = 12
    n = 1 << m
    batch_size = 512
    rank = 3
    subdiag = Variable(torch.rand(n-1), requires_grad=True).cuda()
    A = np.diag(subdiag.data.cpu().numpy(), -1)
    u = Variable(torch.rand((batch_size, n)), requires_grad=True).cuda()
    v = Variable(torch.rand((rank, n)), requires_grad=True).cuda()
    w = Variable(torch.rand((batch_size, rank, n)), requires_grad=True).cuda()
    result_fast = krylov_multiply_fast(subdiag, v, w)
    grad_fast,  = torch.autograd.grad(torch.sum(result_fast), v, retain_graph=True)
    grad_fast = grad_fast.data.cpu().numpy()
    result_fast = result_fast.data.cpu().numpy()
    result = krylov_multiply(subdiag, v, w)
    grad, = torch.autograd.grad(torch.sum(result), v, retain_graph=True)
    grad = grad.data.cpu().numpy()
    result = result.data.cpu().numpy()
    # Using autodiff
    result1 = krylov_multiply_by_autodiff(subdiag, v, w)
    result1 = result1.data.cpu().numpy()
    # CPU dense multiply
    Ks = [krylov_construct(A, v.data.cpu().numpy()[i], n) for i in range(rank)]
    w_cpu = w.data.cpu().numpy()
    result2 = np.stack([w_cpu[:, i] @ Ks[i] for i in range(rank)]).sum(axis=0)
    result2 = result2.squeeze()
    np.allclose(result_fast, result)
    np.allclose(result, result1)
    np.allclose(result1, result2)
    print(np.max(abs(result_fast - result)))
    print(np.mean(abs(result_fast - result)))
    print(np.max(abs(grad_fast - grad)))
    print(np.mean(abs(grad_fast - grad)))
    print(np.max(abs(result - result1)))
    print(np.mean(abs(result - result1)))
    print(np.max(abs(result1 - result2)))
    print(np.mean(abs(result1 - result2)))

    # Combine transpose multiply follow by non-transpose multiply
    result = krylov_multiply(subdiag, v, krylov_transpose_multiply(subdiag, v, u))

def test_misc():
    a = Variable(torch.rand(3, 4, 8).cuda(), requires_grad=True)
    # b_re, b_im = fft.autograd.Fft(a)
    b_re, b_im = fft.rfft(a.data)

    b = b_re.cpu().numpy() + 1j * b_im.cpu().numpy()
    b_np = np.fft.rfft(a.cpu().numpy())
    np.allclose(b, b_np)

    temp = Variable(torch.zeros_like(a.data))
    temp[0] = a[0]
    s = temp.sum()
    from torch import autograd
    g = autograd.grad(s, a)

    u = Variable(torch.rand((1, 8)), requires_grad=True).cuda()
    re, im = Rfft(u)
    t = Irfft(re, im)
    t1_re, t1_im = Rfft(t)

    grad, = torch.autograd.grad(torch.sum(re), u, retain_graph=True)
    w = Variable(torch.rand((1, 5)), requires_grad=True).cuda()
    ggrad = torch.autograd.grad(torch.sum(grad), u)
    torch.autograd.gradcheck(Rfft, (u, ))

    analytic_grad = fft.irfft(torch.ones_like(re).data, -torch.ones_like(im).data, normalize=False)

    epsilon = 1e-5
    for i in range(2):
        one_hot = Variable(torch.zeros_like(u.data))
        one_hot[0, i] = epsilon
        u_new = u + one_hot
        u_new_minus = u - one_hot
        print((torch.sum(Rfft(u_new)[0]) - torch.sum(Rfft(u_new_minus)[0])) / (2 * epsilon))

def shift(v, f=1):
    return torch.cat((f * v[[v.size(0) - 1]], v[:-1]))

def shift_subdiag(subdiag, v, f=0.0):
    return torch.cat((f * v[[v.size(0) - 1]], subdiag * v[:-1]))

def Krylov(linear_map, v, n=None):
    if n is None:
        n = v.size(0)
    cols = [v]
    for _ in range(n - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)


if __name__ == "__main__":
    test_tranpose_multiply()
    test_multiply()
