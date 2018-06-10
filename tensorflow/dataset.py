from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os,sys,h5py
import scipy.io as sio
from scipy.linalg import solve_sylvester
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
# sys.path.insert(0, '../../../../')
from utils import *

class Dataset:
    # here n is the input size.
    # true_test: if True, we test on test set. Otherwise, split training set into train/validation.
    def __init__(self, name, layer_size, num_iters, transform, stochastic_train, replacement, test_size=1000, train_size=10000, true_test=False, train_fraction=1.0, val_fraction=0.15):
        self.name = name
        self.mnist = None
        # train_fraction and val_fraction used only in sample complexity experiments currently
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.transform = transform
        self.replacement = replacement
        self.stochastic_train = stochastic_train
        self.num_iters = num_iters
        self.layer_size = layer_size
        self.pert = None
        self.current_batch = 0
        self.true_transform = None
        self.test_size = test_size
        self.train_size = train_size
        self.true_test = true_test
        self.input_size = self.get_input_size()
        self.set_data_locs()

        if self.name in ['iwslt', 'copy']:
            return
        elif self.name == 'timit':
            train_feat_loc = '../../../timit/timit_train_feat.mat'
            train_lab_loc = '../../../timit/timit_train_lab.mat'
            train_X = h5py.File(train_feat_loc, 'r')['fea']
            print('loaded')
            train_X = np.array(train_X).T
            print('train_X: ', train_X.shape)
            train_Y = sio.loadmat(train_lab_loc)['lab']
            # Ensure Y is one-hot
            enc = OneHotEncoder()
            train_Y = enc.fit_transform(train_Y).todense()
            # Split into validation and train
            val_size = int(0.1*train_X.shape[0]) 
            train_size = train_X.shape[0] - val_size
            # Shuffle X
            idx = np.arange(0, train_X.shape[0])
            np.random.shuffle(idx)

            train_idx = idx[0:train_size]
            val_idx = idx[-val_size:]

            assert train_idx.size == train_size
            assert val_idx.size == val_size

            self.val_X = train_X[val_idx, :]
            self.val_Y = train_Y[val_idx, :]
            self.train_X = train_X[train_idx, :]
            self.train_Y = train_Y[train_idx, :]
        elif self.name.startswith('mnist_noise') or self.name in ['mnist_bg_rot', 'convex', 'rect', 'norb', 'norb_val','cifar10']:
            data = pkl.load(open(self.train_loc, 'rb'))
            train_X = data['X']
            train_Y = data['Y']

            # Shuffle
            idx = np.arange(train_X.shape[0])
            np.random.shuffle(idx)
            train_X = train_X[idx,:]
            train_Y = train_Y[idx,:]

            # Downsample for sample complexity experiments
            if self.train_fraction is not None:
                num_samples = int(self.train_fraction*train_X.shape[0])
                train_X = train_X[0:num_samples,:]
                train_Y = train_Y[0:num_samples,:]

                val_size = int(self.val_fraction*train_X.shape[0])

            else:
                if self.name == 'norb_val':
                    val_size = 50000
                elif self.name == 'rect':
                    val_size = 100
                elif self.name == 'convex':
                    val_size = 800
                else:
                    val_size = 2000
            train_size = train_X.shape[0] - val_size
            # Shuffle X
            idx = np.arange(0, train_X.shape[0])
            np.random.shuffle(idx)

            train_idx = idx[0:train_size]
            val_idx = idx[-val_size:]

            assert train_idx.size == train_size
            assert val_idx.size == val_size

            self.val_X = train_X[val_idx, :]
            self.val_Y = train_Y[val_idx, :]
            self.train_X = train_X[train_idx, :]
            self.train_Y = train_Y[train_idx, :]

            # post-processing transforms
            self.train_X, _ = self.postprocess(self.train_X)
            self.val_X, _ = self.postprocess(self.val_X)

        elif self.name == 'smallnorb':
            data_loc = '/dfs/scratch1/thomasat/datasets/smallnorb/processed_py2.pkl'
            # Load
            data = pkl.load(open(data_loc, 'rb'))
            train_X = data['train_X']
            train_Y = data['train_Y']
            self.test_X = data['test_X']
            self.test_Y = data['test_Y']
            val_size = 2000
            train_size = train_X.shape[0] - val_size

            print(('train size, val size: ', train_size, val_size))

            # Shuffle X
            idx = np.arange(0, train_X.shape[0])
            np.random.shuffle(idx)

            train_idx = idx[0:train_size]
            val_idx = idx[-val_size:]

            assert train_idx.size == train_size
            assert val_idx.size == val_size

            self.val_X = train_X[val_idx, :]
            self.val_Y = train_Y[val_idx, :]
            self.train_X = train_X[train_idx, :]
            self.train_Y = train_Y[train_idx, :]
        elif self.name == 'mnist':
            data_dir = '/tmp/tensorflow/mnist/input_data'
            self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
            self.train_X = self.mnist.train.images
            self.train_Y = self.mnist.train.labels
            self.val_X = self.mnist.validation.images
            self.val_Y = self.mnist.validation.labels
            self.test_X = self.mnist.test.images
            self.test_Y = self.mnist.test.labels
            # postprocess
            self.train_X, self.train_Y = self.augment(self.train_X, self.train_Y)
            # self.test_X, self.test_Y = self.augment(self.test_X, self.test_Y)
        elif self.name == 'mnist_rot':
            self.load_train_data()
        elif self.name in ['swap_mnist_bg_rot']:
            if self.name.startswith('swap'):
                train_size = 40000
                val_size = 10000
            else:
                train_size = 10000
                val_size = 2000
            data = np.genfromtxt(self.train_loc)

            # Shuffle
            X = data[:, :-1]
            Y = np.expand_dims(data[:, -1], 1)

            # Y must be one-hot
            enc = OneHotEncoder()
            Y = enc.fit_transform(Y).todense()
            idx = np.arange(0, X.shape[0])
            np.random.shuffle(idx)

            train_idx = idx[0:train_size]
            val_idx = idx[-val_size:]

            assert train_idx.size == train_size
            assert val_idx.size == val_size

            self.train_X = X[train_idx, :]
            self.train_Y = Y[train_idx, :]
            self.val_X = X[val_idx, :]
            self.val_Y = Y[val_idx, :]

            # post-processing transforms
            self.train_X, _ = self.postprocess(self.train_X)
            self.val_X, _ = self.postprocess(self.val_X)
        elif self.name == 'mnist_rand_bg':
            self.load_train_data()
        elif self.name == 'rect_images':
            train_size = 11000
            val_size = 1000
            data = np.genfromtxt(self.train_loc)

            # Shuffle
            X = data[:, :-1]
            Y = np.expand_dims(data[:, -1], 1)

            # Y must be one-hot
            enc = OneHotEncoder()
            Y = enc.fit_transform(Y).todense()
            idx = np.arange(0, X.shape[0])
            np.random.shuffle(idx)

            train_idx = idx[0:train_size]
            val_idx = idx[-val_size:]

            assert train_idx.size == train_size
            assert val_idx.size == val_size

            self.train_X = X[train_idx, :]
            self.train_Y = Y[train_idx, :]
            self.val_X = X[val_idx, :]
            self.val_Y = Y[val_idx, :]
        elif self.name.startswith('true'):
            self.true_transform = gen_matrix(self.input_size, self.name.split("true_",1)[1] )
            test_X, test_Y = gen_batch(self.true_transform, self.test_size)
            val_X, val_Y = gen_batch(self.true_transform, self.test_size)
            self.test_X = test_X
            self.test_Y = test_Y
            self.val_X = val_X
            self.val_Y = val_Y

            if not self.stochastic_train:
                train_X, train_Y = gen_batch(self.true_transform, self.train_size)
                self.train_X = train_X
                self.train_Y = train_Y
        else:
            print('Not supported: ', self.name)
            assert 0



        if not self.replacement:
            #For batching
            self.current_idx = 0


        print('Training set X,Y: ', self.train_X.shape, self.train_Y.shape)
        print('Validation set X,Y: ', self.val_X.shape, self.val_Y.shape)
        # self.print_dataset_stats()

    def print_dataset_stats(self,test=False):
        print('Train X mean, std: ', np.mean(self.train_X,axis=0), np.std(self.train_X,axis=0))
        print('Train X min, max: ', np.min(self.train_X), np.max(self.train_X))
        print('Val X mean, std: ', np.mean(self.val_X,axis=0), np.std(self.val_X,axis=0))
        print('Val X min, max: ', np.min(self.val_X), np.max(self.val_X))

        if test:
            print('Test X mean, std: ', np.mean(self.test_X,axis=0), np.std(self.test_X,axis=0))
            print('Test X min, max: ', np.min(self.test_X), np.max(self.test_X))

    def set_data_locs(self):
        prefix = '/dfs/scratch1/thomasat/datasets/'
        if self.name == 'cifar10':
            data_dir = prefix + 'cifar10_combined'
            train_name = 'train_grayscale' if 'grayscale' in self.transform else 'train'
            test_name = 'test_grayscale' if 'grayscale' in self.transform else 'test'
            self.train_loc = os.path.join(data_dir, train_name)
            self.test_loc = os.path.join(data_dir, test_name)
        elif self.name.startswith('mnist_noise'):
            idx = self.name[-1]
            self.train_loc = os.path.join(prefix,'mnist_noise/train_' + str(idx))
            self.test_loc = os.path.join(prefix,'mnist_noise/test_' + str(idx))
        elif self.name.startswith('swap_mnist_noise'):
            idx = self.name[-1]
            self.train_loc = os.path.join(prefix,'mnist_noise/test_' + str(idx))
            self.test_loc = os.path.join(prefix,'mnist_noise/train_' + str(idx))
        elif self.name == 'norb' or self.name=='norb_val':
            self.train_loc = os.path.join(prefix,'norb_full/processed_py2_train_28.pkl')
            self.test_loc = os.path.join(prefix,'norb_full/processed_py2_test_28.pkl')
        elif self.name == 'rect_images':
            self.train_loc = os.path.join(prefix, 'rect_images/rectangles_im_train.amat')
            self.test_loc = os.path.join(prefix, 'rect_images/rectangles_im_test.amat')
        elif self.name == 'rect':
            self.train_loc = os.path.join(prefix,'rect/train_normalized')
            self.test_loc = os.path.join(prefix, 'rect/test_normalized')
        elif self.name == 'convex':
            self.train_loc = os.path.join(prefix, 'convex/train_normalized')
            self.test_loc = os.path.join(prefix, 'convex/test_normalized')
        elif self.name == 'mnist_rand_bg':
            self.train_loc = os.path.join(prefix, 'mnist_rand_bg/mnist_background_random_train.amat')
            self.test_loc = os.path.join(prefix, 'mnist_rand_bg/mnist_background_random_test.amat')
        elif self.name == 'mnist_bg_rot':
            self.train_loc = os.path.join(prefix, 'mnist_bg_rot/train_normalized')
            self.test_loc = os.path.join(prefix, 'mnist_bg_rot/test_normalized')
        elif self.name == 'swap_mnist_bg_rot':
            self.train_loc = os.path.join(prefix, 'mnist_bg_rot/mnist_all_background_images_rotation_normalized_test.amat')
            self.test_loc = os.path.join(prefix, 'mnist_bg_rot/mnist_all_background_images_rotation_normalized_train_valid.amat')

    def get_input_size(self):
        if 'mnist' in self.name or 'convex' in self.name or 'rect' in self.name:
            if 'pad' in self.transform:
                return 1024
            else:
                return 784
        elif self.name == 'smallnorb':
            return 576
        elif self.name == 'norb' or self.name=='norb_val':
            return 784#729
        elif self.name == 'timit':
            return 440
        elif self.name == 'cifar10':
            if 'grayscale' in self.transform:
                return 1024
            elif 'downsample' in self.transform:
                return 768
            return 3072
        elif self.name.startswith('true_') or self.name in ['iwslt', 'copy']:
            return self.layer_size
        else:
            print('Name not recognized: ', name)
            assert 0

    def postprocess(self, X, Y=None):
        # pad from 784 to 1024
        if 'pad' in self.transform:
            X = np.pad(X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
            # self.train_X = np.pad(self.train_X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
            # self.val_X = np.pad(self.val_X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
            # self.test_X = np.pad(self.test_X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
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

    def out_size(self):
        if self.name in ['convex', 'rect', 'rect_images']:
            return 2
        elif self.name == 'smallnorb':
            return 5
        elif self.name == 'norb':
            return 6
        elif 'mnist' in self.name or 'cifar10' in self.name:
            return 10
        elif self.name == 'timit':
            return 147
        else:
            return self.input_size

    def load_test_data(self):
        if self.name == 'mnist':
            pass
        elif self.name == 'timit':
            test_feat_loc = '../../../timit/timit_heldout_feat.mat'
            test_lab_loc = '../../../timit/timit_heldout_lab.mat'
            test_X = sio.loadmat(test_feat_loc)['fea']
            print('loaded test')
            test_X = np.array(test_X)
            print('test_X: ', test_X.shape)
            test_Y = sio.loadmat(test_lab_loc)['lab']
            # Ensure Y is one-hot
            enc = OneHotEncoder()
            test_Y = enc.fit_transform(test_Y).todense()
            self.test_X = test_X
            self.test_Y = test_Y
            print('test_Y: ', test_Y.shape)
        elif self.name == 'smallnorb':
            return
        elif self.name.startswith('mnist_noise') or self.name in ['norb', 'norb_val','cifar10', 'convex', 'rect', 'mnist_bg_rot']:
            data = pkl.load(open(self.test_loc, 'rb'))
            self.test_X = data['X']
            self.test_Y = data['Y']
        elif self.test_loc:
            test_data = np.genfromtxt(self.test_loc)

            self.test_X = test_data[:, :-1]
            self.test_Y = np.expand_dims(test_data[:, -1], 1)

            # Y must be one-hot
            enc = OneHotEncoder()
            self.test_Y = enc.fit_transform(self.test_Y).todense()
        self.test_X, _ = self.postprocess(self.test_X)
        print('Loaded test data: ')
        print('Test X,Y:', self.test_X.shape, self.test_Y.shape)
        # self.print_dataset_stats(test=True)

    def load_train_data(self):
        train_data = np.genfromtxt(self.train_loc)

        self.train_X = train_data[:, :-1]
        self.train_Y = np.expand_dims(train_data[:, -1], 1)

        # Y must be one-hot
        enc = OneHotEncoder()
        self.train_Y = enc.fit_transform(self.train_Y).todense()

    def update_batch_idx(self, batch_size):
        self.current_idx += batch_size
        if self.current_idx >= self.train_X.shape[0]:
            self.current_idx = 0
        #print('Current training data index: ', self.current_idx)

    def next_batch(self, batch_size):
        #Randomly shuffle training set at the start of each epoch if sampling without replacement
        if self.current_idx == 0:
            idx = np.arange(0, self.train_X.shape[0])
            np.random.shuffle(idx)
            self.train_X = self.train_X[idx,:]
            self.train_Y = self.train_Y[idx,:]
            print('Shuffling: new epoch')

        idx_end = min(self.train_X.shape[0], self.current_idx+batch_size)
        batch_X = self.train_X[self.current_idx:idx_end,:]
        batch_Y = self.train_Y[self.current_idx:idx_end,:]
        self.update_batch_idx(batch_size)
        return batch_X, batch_Y

    def batch(self, batch_size, step):
        if self.replacement:
            return self.sample_with_replacement(batch_size, step)
        else:
            return self.sample_without_replacement(batch_size, step)

    def sample_with_replacement(self, batch_size, step):
        # if self.name == 'mnist':
        #     batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
        #     return batch_xs, batch_ys
        if self.name.startswith('mnist') \
             or self.name.startswith('swap_mnist') \
             or self.name in ['convex', 'rect', 'rect_images', 'smallnorb', 'norb', 'norb_val', 'cifar10', 'timit']:
            #Randomly sample batch_size from train_X and train_Y
            idx = np.random.randint(self.train_X.shape[0], size=batch_size)
            return self.train_X[idx, :], self.train_Y[idx, :]
        elif self.name.startswith('true'):
            if self.stochastic_train:
                return gen_batch(self.true_transform, batch_size, self.pert)
            else:
                return self.next_batch(batch_size)
        else:
            print('Not supported: ', self.name)
            assert 0

    def sample_without_replacement(self, batch_size, step):
        # if self.name == 'mnist':
        #     batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
        #     return batch_xs, batch_ys
        # elif self.name.startswith('mnist') or self.name in ['convex', 'rect', 'rect_images', 'smallnorb', 'norb', 'cifar10']:
        if self.name.startswith('mnist') \
             or self.name.startswith('swap_mnist') \
             or self.name in ['convex', 'rect', 'rect_images', 'smallnorb', 'norb', 'norb_val', 'cifar10', 'timit']:
            return self.next_batch(batch_size)
        elif self.name.startswith('true'):
            if self.stochastic_train:
                return gen_batch(self.true_transform, batch_size, self.pert)
            else:
                return self.next_batch(batch_size)
        else:
            print('Not supported: ', name)
            assert 0
