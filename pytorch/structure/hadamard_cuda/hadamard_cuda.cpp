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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard_transform", &hadamard_transform, "Fast Hadamard transform");
}
