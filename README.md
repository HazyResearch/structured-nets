# structured-nets
Code to accompany the paper <a href="https://arxiv.org/abs/1810.02309">Learning Compressed Transforms with Low Displacement Rank</a>.

## Requirements
Python 3.6+

PyTorch >=0.4.1

## Installing CUDA Extensions
Some functions are written in CUDA for speed. To install them:
```
cd pytorch/structure/hadamard_cuda
python setup.py install

cd pytorch/structure/diag_mult_cuda
python setup.py install
```


## Example Usage

See <a href="https://github.com/HazyResearch/structured-nets/tree/master/pytorch" rel="nofollow">our PyTorch section</a>.
