# Copyright 2018 HazyResearch
# https://github.com/HazyResearch/structured-nets
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from math import log2

import hadamard_cuda
import torch


def hadamard_transform_torch(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    batch_size, n = u.shape
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"
    x = u[..., None]
    for _ in range(m):
        top = x[..., ::2, :] + x[..., 1::2, :]
        bot = x[..., ::2, :] - x[..., 1::2, :]
        x = torch.cat((top, bot), dim=-1)
    return x.squeeze(-2) / 2 ** (m / 2) if normalize else x.squeeze(-2)


class HadamardTransformCuda(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(ctx, u):
        return hadamard_cuda.hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return HadamardTransformCuda.apply(grad)


def hadamard_transform_cuda(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    _, n = u.shape
    m = int(log2(n))
    assert n == 1 << m, "n must be a power of 2"
    output = HadamardTransformCuda.apply(u)
    return output / 2 ** (m / 2) if normalize else output
