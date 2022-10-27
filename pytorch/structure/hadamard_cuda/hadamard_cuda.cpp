/*
Copyright 2018 HazyResearch
https://github.com/HazyResearch/structured-nets

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <torch/extension.h>

void fwtBatchGPU(float* x, int batchSize, int log2N);

torch::Tensor hadamard_transform(torch::Tensor x) {
  TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
  auto n = x.size(-1);
  auto log2N = long(log2(n));
  TORCH_CHECK(n == 1 << log2N, "n must be a power of 2");
  auto output = x.clone();  // Cloning makes it contiguous.
  auto batchSize = x.numel() / (1 << log2N);
  fwtBatchGPU(output.data_ptr<float>(), batchSize, log2N);
  return output;
}

PYBIND11_MODULE(HADAMARD_EXTENSION_NAME, m) {
  m.def("hadamard_transform", &hadamard_transform, "Fast Hadamard transform");
}

TORCH_LIBRARY(HADAMARD_EXTENSION_NAME, m) {
  m.def("hadamard_transform", &hadamard_transform);
}
