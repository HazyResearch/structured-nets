#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_hadamard_transform(py::module &);
void init_diag_mult(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_hadamard_transform(m);
  init_diag_mult(m);
}
