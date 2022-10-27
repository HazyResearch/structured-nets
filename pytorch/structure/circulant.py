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


import torch


def circulant_multiply(c, x):
    """Multiply circulant matrix with first column c by x.
    E.g. if the matrix is
        [1 2 3]
        [2 3 1]
        [3 1 2]
    c should be [1,2,3]
    Parameters:
        c: (n, )
        x: (batch_size, n) or (n, )
    Return:
        prod: (batch_size, n) or (n, )
    """
    return torch.fft.irfft(torch.fft.rfft(c) * torch.fft.rfft(x))
