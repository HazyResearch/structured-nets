from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os,sys,h5py
import scipy.io as sio
from scipy.linalg import solve_sylvester
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
import torch
from torchvision import datasets, transforms

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataset(dataset_name):
    """
    Get dataset specific information: the actual data, validation split size, input/output sizes
    """
    # if dataset_name == 'mnist':
    #     mnist_normalize = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307, ), (0.3081, ))
    #     ])
    #     mnist_train = datasets.MNIST(
    #         '../data', train=True, download=True, transform=mnist_normalize)
    #     mnist_test = datasets.MNIST(
    #         '../data', train=False, download=True, transform=mnist_normalize)
    #     val_size = 5000
    #     in_size = 784
    #     out_size = 10

    #     train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=len(mnist_train), shuffle=True)
    #     for X, Y in train_loader:
    #         train_X, train_Y = X, Y
    #     test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=len(mnist_test), shuffle=True)
    #     for X, Y in test_loader:
    #         test_X, test_Y = X, Y
    #     train_X = train_X.view(-1, in_size)
    #     test_X = test_X.view(-1, in_size)
    #     train_Y = torch.zeros(train_X.size(0), out_size)
    #     train_Y.scatter_(1, mnist_train.train_labels.unsqueeze(1), 1)
    #     # test_X = mnist_test.test_data.view(-1, in_size)
    #     test_Y = torch.zeros(test_X.size(0), out_size)
    #     test_Y.scatter_(1, mnist_test.test_labels.unsqueeze(1), 1)
    #     # return train_X, train_Y, test_X, test_Y, val_size, in_size, out_size
    #     return torch.FloatTensor(train_X), torch.FloatTensor(train_Y), torch.FloatTensor(test_X), torch.FloatTensor(test_Y), val_size, in_size, out_size
    #     # normalize

    prefix = '/dfs/scratch1/thomasat/datasets/'
    if dataset_name == 'mnist':
        train_loc = os.path.join(prefix, 'mnist/train_normalized')
        test_loc = os.path.join(prefix, 'mnist/test_normalized')
        val_size = 5000
        out_size = 10
    elif dataset_name == 'cifar10':
        train_loc = os.path.join(prefix, 'cifar10_combined/train')
        test_loc = os.path.join(prefix, 'cifar10_combined/test')
        val_size = 5000
        out_size = 10
    elif dataset_name == 'cifar10mono':
        train_loc = os.path.join(prefix, 'cifar10_combined/train_grayscale')
        test_loc = os.path.join(prefix, 'cifar10_combined/test_grayscale')
        val_size = 5000
        out_size = 10
    elif dataset_name.startswith('mnist_noise'):
        idx = dataset_name[-1]
        train_loc = os.path.join(prefix,'mnist_noise/train_' + str(idx))
        test_loc = os.path.join(prefix,'mnist_noise/test_' + str(idx))
        val_size = 2000
        out_size = 10
    elif dataset_name == 'norb':
        train_loc = os.path.join(prefix,'norb_full/processed_py2_train_28.pkl')
        test_loc = os.path.join(prefix,'norb_full/processed_py2_test_28.pkl')
        val_size = 30000
        out_size = 6
    elif dataset_name == 'rect_images': #TODO
        train_loc = os.path.join(prefix, 'rect_images/rectangles_im_train.amat')
        test_loc = os.path.join(prefix, 'rect_images/rectangles_im_test.amat')
        out_size = 2
    elif dataset_name == 'rect':
        train_loc = os.path.join(prefix,'rect/train_normalized')
        test_loc = os.path.join(prefix, 'rect/test_normalized')
        val_size = 100
        out_size = 2
    elif dataset_name == 'convex':
        train_loc = os.path.join(prefix, 'convex/train_normalized')
        test_loc = os.path.join(prefix, 'convex/test_normalized')
        val_size = 800
        out_size = 2
    elif dataset_name == 'mnist_rand_bg': #TODO
        train_loc = os.path.join(prefix, 'mnist_rand_bg/mnist_background_random_train.amat')
        test_loc = os.path.join(prefix, 'mnist_rand_bg/mnist_background_random_test.amat')
        val_size = 2000
        out_size = 10
    elif dataset_name == 'mnist_bg_rot':
        train_loc = os.path.join(prefix, 'mnist_bg_rot/train_normalized')
        test_loc = os.path.join(prefix, 'mnist_bg_rot/test_normalized')
        val_size = 2000
        out_size = 10
    elif dataset_name == 'mnist_bg_rot_swap':
        train_loc = os.path.join(prefix, 'mnist_bg_rot/test_normalized')
        test_loc = os.path.join(prefix, 'mnist_bg_rot/train_normalized')
        val_size = 5000
        out_size = 10
    elif dataset_name == 'timit': #TODO
        out_size = 147
    #TODO handle iwslt, copy tasks
    # TODO smallnorb
    else:
        print('dataset.py: unknown dataset name')

    # TODO maybe want the .amat if that's standard and do postprocessing in a uniform way instead of having a separate script per dataset
    train_data = pkl.load(open(train_loc, 'rb'))
    train_X = train_data['X']
    train_Y = train_data['Y']
    test_data = pkl.load(open(test_loc, 'rb'))
    test_X = test_data['X']
    test_Y = test_data['Y']
    in_size = train_X.shape[1]

    print("Train dataset size: ", train_X.shape[0])
    print("Test dataset size: ", test_X.shape[0])

    return torch.FloatTensor(train_X), torch.FloatTensor(train_Y), torch.FloatTensor(test_X), torch.FloatTensor(test_Y), val_size, in_size, out_size

def split_train_val(train_X, train_Y, val_size, train_fraction=None, val_fraction=None):
    """
    Input: training data as a torch.Tensor
    """
    # Shuffle
    idx = np.arange(train_X.shape[0])
    np.random.shuffle(idx)
    train_X = train_X[idx,:]
    train_Y = train_Y[idx,:]

    # Downsample for sample complexity experiments
    if val_fraction is not None:
        val_size = int(val_fraction*train_X.shape[0])
    if train_fraction is not None:
        train_size = int(train_fraction*train_X.shape[0])
        assert val_size + train_size <= train_X.shape[0]
    else:
        train_size = train_X.shape[0] - val_size

    # Shuffle X
    idx = np.arange(0, train_X.shape[0])
    np.random.shuffle(idx)

    train_idx = idx[0:train_size]
    val_idx = idx[-val_size:]
    val_X = train_X[val_idx, :]
    val_Y = train_Y[val_idx, :]
    train_X = train_X[train_idx, :]
    train_Y = train_Y[train_idx, :]
    return train_X, train_Y, val_X, val_Y



def create_data_loaders(dataset_name, transform, train_fraction, val_fraction, batch_size):
    if device.type == 'cuda':
        loader_args = {'num_workers': 16, 'pin_memory': True}
    else:
        loader_args = {'num_workers': 4, 'pin_memory': False}

    train_X, train_Y, test_X, test_Y, val_size, in_size, out_size = get_dataset(dataset_name) # train/test data, val size, input/output size
    # train_X, train_Y = postprocess(transform, train_X, train_Y)
    # test_X, test_Y = postprocess(transform, test_X, test_Y)

    # TODO: use torch.utils.data.random_split instead
    # however, this requires creating the dataset, then splitting, then applying transformations
    train_X, train_Y, val_X, val_Y = split_train_val(train_X, train_Y, val_size, train_fraction, val_fraction)


    # TODO: use pytorch transforms to postprocess

    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_Y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    return train_loader, val_loader, test_loader, in_size, out_size


class DatasetLoaders:
    def __init__(self, name, transform=None, train_fraction=None, val_fraction=None, batch_size=50):
        if name.startswith('true'):
            # TODO: Add support for synthetic datasets back. Possibly should be split into separate class
            self.loss = utils.mse_loss
        else:
            self.train_loader, self.val_loader, self.test_loader, self.in_size, self.out_size = create_data_loaders(name, transform, train_fraction, val_fraction, batch_size)
            self.loss = utils.cross_entropy_loss





### Utilities for processing data arrays in numpy
def postprocess(transform, X, Y=None):
    # pad from 784 to 1024
    if 'pad' in transform:
        assert X.shape[1] == 784
        print(X.shape, type(X))
        X = np.pad(X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
    return X, Y

def augment(self, X, Y=None):
    if 'contrast' in self.transform:
        def scale_patch(X):
            patch = ((9, 19), (9, 19))
            X_ = X.copy()
            X_[:, patch[0][0]:patch[0][1], patch[1][0]:patch[1][1]] *= 2
            return X_
        # subsample
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X[idx,...]
        Y = Y[idx,...]

        X1 = X.reshape((-1,28,28))
        X2 = scale_patch(X1)
        X3 = scale_patch(X2)
        X4 = scale_patch(X3)
        # X5 = scale_patch(X4)
        X = np.concatenate([X1, X2, X3, X4], axis=0).reshape(-1, 28*28)
        Y = np.concatenate([Y, Y, Y, Y], axis=0)

    if 'patch' in self.transform:
        def add_patch(X):
            patch = ((0, 4), (10, 18))
            X_ = X.copy()
            X_[:, patch[0][0]:patch[0][1], patch[1][0]:patch[1][1]] += 3.0
            return X_
        X1 = X.reshape((-1,28,28))
        X2 = add_patch(X1)
        X3 = add_patch(X2)
        X4 = add_patch(X3)
        X = np.concatenate([X1, X2, X3, X4], axis=0).reshape(-1, 28*28)
        Y = np.concatenate([Y, Y, Y, Y], axis=0)

    return X, Y