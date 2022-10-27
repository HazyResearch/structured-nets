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


from .hadamard import hadamard_transform_cuda, hadamard_transform_torch

# S,G,B: diagonal
# P: permutation
# x: batch_size x n_features
def fastfood_multiply(S, G, B, P, x):
    HBx = hadamard_transform_torch(B * x)
    PHBx = HBx[:, P]
    HGPHBx = hadamard_transform_torch(G * PHBx)
    return S * HGPHBx


def fastfood_multiply_cuda(S, G, B, P, x):
    HBx = hadamard_transform_cuda(B * x)
    PHBx = HBx[:, P]
    HGPHBx = hadamard_transform_cuda(G * PHBx)
    return S * HGPHBx
