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

void subdiagMultGPU(
    float* d_Subdiag,
    float* d_Data,
    float* d_Output,
    int shiftSubdiag,
    int shiftV,
    int batchSize,
    int N,
    bool batchedSubdiag);

torch::Tensor cycle_mult(
    torch::Tensor subdiag,
    torch::Tensor v,
    int64_t shiftSubdiag,
    int64_t shiftV) {
  TORCH_CHECK(subdiag.is_cuda(), "subdiag must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  // Need to make tensors contiguous before passing to CUDA
  subdiag = subdiag.contiguous();
  v = v.contiguous();
  auto n = v.sizes().back();
  auto batchSize = v.numel() / n;
  auto output = torch::empty_like(v);
  bool batchedSubdiag = subdiag.numel() == v.numel();
  subdiagMultGPU(
      subdiag.data_ptr<float>(),
      v.data_ptr<float>(),
      output.data_ptr<float>(),
      shiftSubdiag,
      shiftV,
      batchSize,
      n,
      batchedSubdiag);
  return output;
}

torch::Tensor subdiagKrylov(torch::Tensor subdiag, torch::Tensor v, int64_t m) {
  TORCH_CHECK(subdiag.is_cuda(), "subdiag must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  // Need to make tensors contiguous before passing to CUDA
  subdiag = subdiag.contiguous();
  v = v.contiguous();
  auto n = v.sizes().back();
  auto batchSize = v.numel() / n;
  auto output = torch::empty(
      {m, batchSize, n}, torch::dtype(v.dtype()).device(v.device()));
  // subdiagKrylovGPU(subdiag.data_ptr<float>(), v.data_ptr<float>(),
  // output.data_ptr<float>(), shiftSubdiag, shiftV, batchSize, n);
  output[0] = v;
  for (int i = 1; i < m; ++i) {
    subdiagMultGPU(
        subdiag.data_ptr<float>(),
        output[i - 1].data_ptr<float>(),
        output[i].data_ptr<float>(),
        0,
        -1,
        batchSize,
        n,
        false);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cycle_mult",
      &cycle_mult,
      "Cycle the vector and then do a pointwise multiplication. Shift should be between -n and n - 1.");
  m.def("subdiagKrylov", &subdiagKrylov, "Subdiag Krylov");
}

TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
  m.def("cycle_mult", &cycle_mult);
  m.def("subdiagKrylov", &subdiagKrylov);
}
