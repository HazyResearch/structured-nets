# PyTorch

`structure/` contains code for matrix multiplication and gradient computation for various structured matrix classes, as well as PyTorch layers for them.

## Example Usage

Example command:
```
python main.py --dataset mnist_noise_1 --result-dir test --lr 1e-3 --epochs 10 model SHL --class-type toeplitz
```
runs a single hidden layer model with the hidden layer constrained to be a Toeplitz-like matrix of equal dimensions to the dataset input size.
The dataset is expected to already be stored at `../../../datasets/{name}`. See `../scripts/data` for example preprocessing scripts, and `models/nets.py` for additional models.

### Flags
- Dataset, training, and optimizer flags are listed with `python main.py -h`
- `model {name}` specifies the end-to-end model {name} corresponding to a class in models/nets.py
- Each model has its own parameters, which can be listed with `python main.py model {name} -h`
- The class-type flag accepts a name of a structured class (e.g. 'toeplitz' or 'subdiagonal\_corner') or an abbreviation (e.g. 't' or 'sdc')

### Multiple Parameters
main.py supports passing in multiple parameters for certain optimizer hyperparameters, and it will search over all combinations. For example,
` python main.py ... --lr 1e-3 2e-3 --mom 0.9 0.99 ... `
will search over 4 combinations of parameters.

For general parameters including model params, this feature can be handled with tools such as xargs or GNU Parallel. E.g.
` parallel python main.py ... model SHL --class-type ::: t sd ::: -r ::: 1 4 16 `
runs Toeplitz-like and LDR subdiagonal ranks 1,4,16.

## Other Tasks

See <a href="https://github.com/thomasat/structured_matrices/tree/master/pytorch/examples" rel="nofollow">here</a> for examples of using a structured layer in additional architectures.

### Butterfly
This repository was used to run experiments for the paper "Learning Structured Transforms with Butterfly Matrices".
Example command:
```
python main.py --dataset cifar10mono --result-dir perm --lr 1e-1 --lr-decay 1.0 --weight-decay 0.0 --trial-id 2 --epochs 100 --batch-size 50 model SHL --class-type b --perm b --depth 3 --real --ortho-init
```
