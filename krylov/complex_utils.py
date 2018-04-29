import torch
import cupy as cp

use_cupy = True

if use_cupy:
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


def conjugate(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


def complex_mult_torch(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def complex_mult_cupy(X, Y):
    '''X and Y are complex64 tensors but stored as torch.cuda.FloatTensor, with last dimension = 2.
    '''
    assert isinstance(X, torch.cuda.FloatTensor) and isinstance(Y, torch.cuda.FloatTensor), 'Input must be torch.cuda.FloatTensor'
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    X_cp, Y_cp = torch_to_cupy(X), torch_to_cupy(Y)
    X_complex, Y_complex = X_cp.view('complex64'), Y_cp.view('complex64')
    out = torch.cuda.FloatTensor(*cp.broadcast(X_cp, Y_cp).shape)
    out_complex = torch_to_cupy(out).view('complex64')
    cp.multiply(X_complex, Y_complex, out_complex)
    return out


class ComplexMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return complex_mult_cupy(X, Y)

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = complex_mult_cupy(grad.data, conjugate(Y)), complex_mult_cupy(grad.data, conjugate(X))
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


complex_mult_ = ComplexMult.apply if use_cupy else complex_mult_torch
