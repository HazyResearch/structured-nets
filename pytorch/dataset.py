import numpy as np
import os,sys,h5py
import scipy.io as sio
from scipy.linalg import solve_sylvester
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F


import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataset(dataset_name, data_dir, transform):
    """
    Get paths of datasets.
    """
    if dataset_name == 'mnist':
        train_loc = os.path.join(data_dir, 'mnist/train_normalized')
        test_loc = os.path.join(data_dir, 'mnist/test_normalized')
    elif dataset_name == 'cifar10':
        train_loc = os.path.join(data_dir, 'cifar10_combined/train')
        test_loc = os.path.join(data_dir, 'cifar10_combined/test')
    elif dataset_name == 'cifar10mono':
        train_loc = os.path.join(data_dir, 'cifar10_combined/train_grayscale')
        test_loc = os.path.join(data_dir, 'cifar10_combined/test_grayscale')
    elif dataset_name.startswith('mnist_noise'):
        idx = dataset_name[-1]
        train_loc = os.path.join(data_dir,'mnist_noise/train_' + str(idx))
        test_loc = os.path.join(data_dir,'mnist_noise/test_' + str(idx))
    elif dataset_name == 'norb':
        train_loc = os.path.join(data_dir,'norb_full/processed_py2_train_32.pkl')
        test_loc = os.path.join(data_dir,'norb_full/processed_py2_test_32.pkl')
    elif dataset_name == 'rect_images': #TODO
        train_loc = os.path.join(data_dir, 'rect_images/rectangles_im_train.amat')
        test_loc = os.path.join(data_dir, 'rect_images/rectangles_im_test.amat')
    elif dataset_name == 'rect':
        train_loc = os.path.join(data_dir,'rect/train_normalized')
        test_loc = os.path.join(data_dir, 'rect/test_normalized')
    elif dataset_name == 'convex':
        train_loc = os.path.join(data_dir, 'convex/train_normalized')
        test_loc = os.path.join(data_dir, 'convex/test_normalized')
    elif dataset_name == 'mnist_rand_bg': #TODO
        train_loc = os.path.join(data_dir, 'mnist_rand_bg/mnist_background_random_train.amat')
        test_loc = os.path.join(data_dir, 'mnist_rand_bg/mnist_background_random_test.amat')
    elif dataset_name == 'mnist_bg_rot':
        train_loc = os.path.join(data_dir, 'mnist_bg_rot/train_normalized')
        test_loc = os.path.join(data_dir, 'mnist_bg_rot/test_normalized')
    elif dataset_name == 'mnist_bg_rot_swap':
        train_loc = os.path.join(data_dir, 'mnist_bg_rot/test_normalized')
        test_loc = os.path.join(data_dir, 'mnist_bg_rot/train_normalized')
    #TODO handle iwslt, copy tasks
    # TODO smallnorb, timit
    else:
        print('dataset.py: unknown dataset name')

    # TODO maybe want the .amat if that's standard and do postprocessing in a uniform way instead of having a separate script per dataset
    train_data = pkl.load(open(train_loc, 'rb'))
    train_X = train_data['X']
    train_Y = train_data['Y']
    test_data = pkl.load(open(test_loc, 'rb'))
    test_X = test_data['X']
    test_Y = test_data['Y']

    train_X, train_Y = postprocess(transform, train_X, train_Y)
    test_X, test_Y = postprocess(transform, test_X, test_Y)

    in_size = train_X.shape[1]
    out_size = train_Y.shape[1]

    print("Train dataset size: ", train_X.shape[0])
    print("Test dataset size: ", test_X.shape[0])
    print("In size: ", in_size)
    print("Out size: ", out_size)

    return torch.FloatTensor(train_X), torch.FloatTensor(train_Y), torch.FloatTensor(test_X), torch.FloatTensor(test_Y), in_size, out_size

def split_train_val(train_X, train_Y, val_fraction, train_fraction=None):
    """
    Input: training data as a torch.Tensor
    """
    # Shuffle
    idx = np.arange(train_X.shape[0])
    np.random.shuffle(idx)
    train_X = train_X[idx,:]
    train_Y = train_Y[idx,:]

    # Compute validation set size
    val_size = int(val_fraction*train_X.shape[0])

    # Downsample for sample complexity experiments
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

    print('train_X: ', train_X.shape)
    print('train_Y: ', train_Y.shape)
    print('val_X: ', val_X.shape)
    print('val_Y: ', val_Y.shape)


    return train_X, train_Y, val_X, val_Y



def create_data_loaders(dataset_name, data_dir, transform, train_fraction, val_fraction, batch_size):
    if device.type == 'cuda':
        loader_args = {'num_workers': 16, 'pin_memory': True}
    else:
        loader_args = {'num_workers': 4, 'pin_memory': False}

    train_X, train_Y, test_X, test_Y, in_size, out_size = get_dataset(dataset_name, data_dir, transform) # train/test data, input/output size
    # train_X, train_Y = postprocess(transform, train_X, train_Y)
    # test_X, test_Y = postprocess(transform, test_X, test_Y)

    # TODO: use torch.utils.data.random_split instead
    # however, this requires creating the dataset, then splitting, then applying transformations
    train_X, train_Y, val_X, val_Y = split_train_val(train_X, train_Y, val_fraction, train_fraction)


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
    def __init__(self, name, data_dir, val_fraction, transform=None, train_fraction=None, batch_size=50):
        if name == 'cifar10':
            loaders, data_shape, self.out_size, val_idx = get_CIFAR10_data('/dfs/scratch1/albertgu/datasets', augment_level=1, batch_size=100,
                     num_val=5000, num_workers=2, seed=None, val_idx=None)
            self.train_loader = loaders['train']
            self.val_loader = loaders['val']
            self.test_loader = loaders['test']
            self.in_size = 3072
            self.loss = utils.torch_cross_entropy_loss
        elif name.startswith('true'):
            # TODO: Add support for synthetic datasets back. Possibly should be split into separate class
            self.loss = utils.mse_loss
        else:
            self.train_loader, self.val_loader, self.test_loader, self.in_size, self.out_size = create_data_loaders(name,
                data_dir, transform, train_fraction, val_fraction, batch_size)
            self.loss = utils.cross_entropy_loss





### Utilities for processing data arrays in numpy
def postprocess(transform, X, Y=None):
    # pad from 784 to 1024
    if 'pad' in transform:
        assert X.shape[1] == 784, X.shape
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

def get_CIFAR10_data(data_dir, augment_level=1, batch_size=100,
                     num_val=5000, num_workers=2, seed=None, val_idx=None):
  """Returns dataloaders and format information for the CIFAR dataset.
     Thanks to https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb."""

  num_train = 50000 - num_val
  if num_train % batch_size:
    log_warn('Batch size does not evenly divide number of training examples')

  normalize = transforms.Normalize(mean=[0.49139765, 0.48215759, 0.44653141],
                                   std=[0.24703199, 0.24348481, 0.26158789])
  test_transform_list = [transforms.ToTensor(), normalize]
  if augment_level == 0:
    train_transform_list = test_transform_list
  elif augment_level == 1:
    train_transform_list = [transforms.ToTensor(),
                            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4,4,4,4), mode='reflect').squeeze()),
                            transforms.ToPILImage(), transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip()] + test_transform_list
  else:
    train_transform_list = [transforms.RandomCrop(size=32, padding=4),
                            transforms.RandomRotation(2.5),
                            transforms.RandomHorizontalFlip()] + \
                            test_transform_list + \
                            [transforms.ColorJitter(.25, .25, .25),
                             Cutout(n_holes=1, length=16)]
  test_transform = transforms.Compose(test_transform_list)
  train_transform = transforms.Compose(train_transform_list)

  dataset_path = data_dir + '/cifar-10-batches-py'
  do_download = not os.path.exists(os.path.expanduser(dataset_path))
  train_set = datasets.CIFAR10(root=data_dir, train=True,
                  download=do_download, transform=train_transform)
  test_set = datasets.CIFAR10(root=data_dir, train=False,
                  download=do_download, transform=test_transform)

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)

  if num_val or val_idx is not None:
    indices = list(range(len(train_set)))
    if val_idx is None:
      np.random.shuffle(indices)
      train_idx, val_idx = indices[num_val:], indices[:num_val]
    else:
      train_idx = np.setdiff1d(indices, val_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    val_set = datasets.CIFAR10(root=data_dir, train=True,
                  download=False, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                     shuffle=False, num_workers=num_workers, sampler=val_sampler,
                     pin_memory=True)
  else:
    train_sampler = None
    val_idx = None
    val_loader = None

  seed_fn = None if seed is None else lambda x: np.random.seed(-(seed+1+x))
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                     shuffle=False, num_workers=num_workers, sampler=train_sampler,
                     worker_init_fn=seed_fn, pin_memory=True)

  data_shape = (3, 32, 32)
  num_classes = 10
  loaders = {'train': train_loader, 'test': test_loader, 'val': val_loader}
  return loaders, data_shape, num_classes, val_idx
