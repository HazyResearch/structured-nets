PyTorch

structure/ contains code for matrix multiplication and gradient computation for various structured matrix classes, as well as PyTorch layers for them


### MLP

Example command:
```
python mlp/main.py --dataset mnist_noise_1 --result_dir test --lr 1e-3 --epochs 10 MLP model SHL --class-type toeplitz

```
runs a single hidden layer model with a Toeplitz-like matrix of equal dimensions to the dataset input size.

Flags:
- Dataset, training, and optimizer flags are listed with `python mlp/main.py -h`
- `MLP model {name}` specifies the end-to-end model {name} corresponding to a class in mlp/nets.py
- Each model has its own parameters, which can be listed with `python mlp/main.py MLP model {name} -h`

### Other tasks

Probably broken right now
