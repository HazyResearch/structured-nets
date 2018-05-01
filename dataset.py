from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os,sys
from scipy.linalg import solve_sylvester
import pickle as pkl
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import OneHotEncoder
sys.path.insert(0, '../../../../')
from utils import *
import torch
from torch.autograd import Variable
sys.path.insert(0, '../../pytorch/')
from torch_utils import *
from torchtext import data

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False).cuda()
        tgt = Variable(data, requires_grad=False).cuda()
        yield Batch(src, tgt, 0)

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class Dataset:
    # here n is the input size.
    # true_test: if True, we test on test set. Otherwise, split training set into train/validation.
    def __init__(self, name, layer_size, num_iters, transform, stochastic_train, replacement, test_size=1000, train_size=10000, true_test=False):
        self.name = name
        self.mnist = None
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
        elif self.name.startswith('mnist_noise') or self.name in ['norb', 'norb_val','cifar10']:
            data = pkl.load(open(self.train_loc, 'rb'))
            train_X = data['X']
            train_Y = data['Y']
            if self.name == 'norb_val':
                val_size = 50000
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
            self.train_X = self.postprocess(self.train_X)
            self.val_X = self.postprocess(self.val_X)
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
            self.val_X = self.mnist.validation.images
            self.val_Y = self.mnist.validation.labels
            self.test_X = self.mnist.test.images
            self.test_Y = self.mnist.test.labels
        elif self.name == 'mnist_rot':
            self.load_train_data()
        elif self.name in ['mnist_bg_rot','swap_mnist_bg_rot']:
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
            self.train_X = self.postprocess(self.train_X)
            self.val_X = self.postprocess(self.val_X)
        elif self.name == 'mnist_rand_bg':
            self.load_train_data()
        elif self.name == 'convex':
            self.load_train_data()
        elif self.name == 'rect':
            train_size = 1100
            val_size = 100
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
        self.print_dataset_stats()

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
            self.train_loc = os.path.join(prefix,'rect/rectangles_train.amat')
            self.test_loc = os.path.join(prefix, 'rect/rectangles_test.amat')
        elif self.name == 'convex':
            self.train_loc = os.path.join(prefix, 'convex/convex_train.amat')
            self.test_loc = os.path.join(prefix, 'convex/50k/convex_test.amat')
        elif self.name == 'mnist_rand_bg':
            self.train_loc = os.path.join(prefix, 'mnist_rand_bg/mnist_background_random_train.amat')
            self.test_loc = os.path.join(prefix, 'mnist_rand_bg/mnist_background_random_test.amat')
        elif self.name == 'mnist_bg_rot':
            self.train_loc = os.path.join(prefix, 'mnist_bg_rot/mnist_all_background_images_rotation_normalized_train_valid.amat')
            self.test_loc = os.path.join(prefix, 'mnist_bg_rot/mnist_all_background_images_rotation_normalized_test.amat')
        elif self.name == 'swap_mnist_bg_rot':
            self.train_loc = os.path.join(prefix, 'mnist_bg_rot/mnist_all_background_images_rotation_normalized_test.amat')
            self.test_loc = os.path.join(prefix, 'mnist_bg_rot/mnist_all_background_images_rotation_normalized_train_valid.amat')

    def get_input_size(self):
        if 'mnist' in self.name or 'convex' in self.name:
            if 'pad' in self.transform:
                return 1024
            else:
                return 784
        elif self.name == 'smallnorb':
            return 576
        elif self.name == 'norb' or self.name=='norb_val':
            return 784#729
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

    def postprocess(self, X):
        # pad from 784 to 1024
        if 'pad' in self.transform:
            X = np.pad(X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
            # self.train_X = np.pad(self.train_X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
            # self.val_X = np.pad(self.val_X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
            # self.test_X = np.pad(self.test_X.reshape((-1,28,28)), ((0,0),(2,2),(2,2)), 'constant').reshape(-1,1024)
        return X

    def out_size(self):
        if self.name in ['convex', 'rect', 'rect_images']:
            return 2
        elif self.name == 'smallnorb':
            return 5
        elif self.name == 'norb':
            return 6
        elif 'mnist' in self.name or 'cifar10' in self.name:
            return 10
        else:
            return self.input_size

    def load_test_data(self):
        if self.name == 'smallnorb':
            return
        elif self.name.startswith('mnist_noise') or self.name in ['norb', 'norb_val','cifar10']:
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
        self.test_X = self.postprocess(self.test_X)
        print('Loaded test data from: ', self.test_loc)
        print('Test X,Y:', self.test_X.shape, self.test_Y.shape)
        self.print_dataset_stats(test=True)

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
        if self.name == 'mnist':
            batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
            return batch_xs, batch_ys
        elif self.name.startswith('mnist') \
             or self.name.startswith('swap_mnist') \
             or self.name in ['convex', 'rect', 'rect_images', 'smallnorb', 'norb', 'norb_val', 'cifar10']:
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
        if self.name == 'mnist':
            batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
            return batch_xs, batch_ys
        # elif self.name.startswith('mnist') or self.name in ['convex', 'rect', 'rect_images', 'smallnorb', 'norb', 'cifar10']:
        elif self.name.startswith('mnist') \
             or self.name.startswith('swap_mnist') \
             or self.name in ['convex', 'rect', 'rect_images', 'smallnorb', 'norb', 'norb_val', 'cifar10']:
            return self.next_batch(batch_size)
        elif self.name.startswith('true'):
            if self.stochastic_train:
                return gen_batch(self.true_transform, batch_size, self.pert)
            else:
                return self.next_batch(batch_size)
        else:
            print('Not supported: ', name)
            assert 0
