#include <torch/torch.h>

void subdiagMultGPU(float *d_Subdiag, float *d_Data, float *d_Output, int shiftSubdiag, int shiftV, int batchSize, int N, bool batchedSubdiag);

at::Tensor cycle_mult(at::Tensor subdiag, at::Tensor v, int shiftSubdiag, int shiftV) {
  AT_CHECK(subdiag.type().is_cuda(), "subdiag must be a CUDA tensor");
  AT_CHECK(v.type().is_cuda(), "v must be a CUDA tensor");
  // Need to make tensors contiguous before passing to CUDA
  subdiag = subdiag.contiguous();
  v = v.contiguous();
  auto n = v.sizes().back();
  auto batchSize = v.numel() / n;
  auto output = at::empty_like(v);
  bool batchedSubdiag = subdiag.numel() == v.numel();
  subdiagMultGPU(subdiag.data<float>(), v.data<float>(), output.data<float>(), shiftSubdiag, shiftV, batchSize, n, batchedSubdiag);
  return output;
}

at::Tensor subdiagKrylov(at::Tensor subdiag, at::Tensor v, int m) {
  AT_CHECK(subdiag.type().is_cuda(), "subdiag must be a CUDA tensor");
  AT_CHECK(v.type().is_cuda(), "v must be a CUDA tensor");
  // Need to make tensors contiguous before passing to CUDA
  subdiag = subdiag.contiguous();
  v = v.contiguous();
  auto n = v.sizes().back();
  auto batchSize = v.numel() / n;
  auto output = at::empty(v.type(), at::IntList{m, batchSize, n});
  // subdiagKrylovGPU(subdiag.data<float>(), v.data<float>(), output.data<float>(), shiftSubdiag, shiftV, batchSize, n);
  output[0] = v;
  for (int i = 1; i < m; ++i) {
    subdiagMultGPU(subdiag.data<float>(), output[i - 1].data<float>(), output[i].data<float>(), 0, -1, batchSize, n, false);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cycle_mult", &cycle_mult, "Cycle the vector and then do a pointwise multiplication. Shift should be between -n and n - 1.");
  m.def("subdiagKrylov", &subdiagKrylov, "Subdiag Krylov");
}
