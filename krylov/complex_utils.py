''' Utility functions for handling complex tensors: conjugate and complex_mult.
Pytorch (as of 0.4.0) does not support complex tensors, so we store them as
float tensors where the last dimension is 2 (real and imaginary parts).
Cupy does support complex arrays, so we can cast Pytorch tensor to Cupy array
and use the multiplication function there. If Cupy isn't available, we use our
own complex multiplication implemented in Pytorch, which is about 6x slower.
'''

import torch

# Check if cupy is available
use_cupy = True
try:
    import cupy as cp
    # Check if cupy works (If it's installed with the wrong version of CUDA, it can be imported but doesn't work)
    a = cp.ndarray(1, 'complex64')
    a = a * a
except:
    use_cupy = False
    print("Cupy isn't installed or isn't working properly. Will use Pytorch's complex multiply, which is much slower.")


if use_cupy:
    CUPY_MEM = cp.ndarray(1, dtype='float32').data.mem


def conjugate(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


def complex_mult_torch(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def torch_to_cupy(tensor):
    '''Hacky way to convert torch.cuda.FloatTensor to CuPy array.
    Probably not safe, since we're manipulating GPU memory addresses directly.
    We will use the official conversion functions once a new CuPy version
    (probably 4.2 or 5.0) is release: https://github.com/cupy/cupy/pull/1082
    '''
    assert isinstance(tensor, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    offset = tensor.data_ptr() - CUPY_MEM.ptr
    array_mem = cp.cuda.memory.MemoryPointer(CUPY_MEM, offset)
    array = cp.ndarray(tensor.shape, dtype='float32', memptr=array_mem)
    array._strides = [4 * s for s in tensor.stride()]
    return array


def complex_mult_cupy_raw(X, Y):
    '''X and Y are complex64 tensors but stored as torch.cuda.FloatTensor, with last dimension = 2.
    Multiply X and Y using Cupy. Operation as implemented is not
    differentiable. Need to use ComplexMultCupy for both the forward and
    backward pass.
    '''
    assert isinstance(X, torch.cuda.FloatTensor) and isinstance(Y, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    X_cp, Y_cp = torch_to_cupy(X), torch_to_cupy(Y)
    X_complex, Y_complex = X_cp.view('complex64'), Y_cp.view('complex64')
    out = torch.cuda.FloatTensor(*cp.broadcast(X_cp, Y_cp).shape)
    out_complex = torch_to_cupy(out).view('complex64')
    cp.multiply(X_complex, Y_complex, out_complex)
    return out


class ComplexMultCupy(torch.autograd.Function):
    '''X and Y are complex64 tensors but stored as torch.cuda.FloatTensor, with last dimension = 2.
    '''
    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return complex_mult_cupy_raw(X, Y)

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = ComplexMultCupy.apply(grad, conjugate(Y)), ComplexMultCupy.apply(grad, conjugate(X))
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
        return grad_X, grad_Y


complex_mult = ComplexMultCupy.apply if use_cupy and torch.cuda.is_available() else complex_mult_torch
