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

void init_diag_mult(py::module &m) {
  m.def("cycle_mult", &cycle_mult, "Cycle the vector and then do a pointwise multiplication. Shift should be between -n and n - 1.");
}
