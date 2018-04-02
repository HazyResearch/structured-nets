# structured_matrices
Learning with structured matrices

## Requirements
Python 3.6+

pytorch>=0.3

pytorch-fft: You need to install with this specific commit. Released version on
PyPI doesn't have Rfft/Irfft that we need and latest version on master is
broken.

`pip install git+git://github.com/locuslab/pytorch_fft.git@c03132a`

cupy: For some reason `pip install cupy` installs the old version 2.5.0 for me,
so you might need `pip install cupy==4.0.0rc1`

