import sys
import pickle as pkl
sys.path.insert(0, '../../')
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from norb import NORBDataset
from scipy.misc import imresize
from data_utils import normalize_data, apply_normalization

MAX_VAL = 255.0
DS_SIZE = (28, 28)
N_CATEGORIES = 6

"""
Downsamples.
"""
def process_image(image):
    # Downsample
    ds = imresize(image, DS_SIZE, 'nearest')

    # Flatten
    return ds.flatten()

"""
Downsamples, stores only left stereo pair, converts to one-hot label.
"""
def process_data(data):
    X = []
    Y = []

    for ex in data:
        this_image = ex.image_lt
        this_category = ex.category
        X.append(process_image(this_image))
        Y.append(this_category)

    X = np.array(X)
    Y = np.array(Y)
    Y = np.expand_dims(Y, 1)
    enc = OneHotEncoder(N_CATEGORIES)
    Y = enc.fit_transform(Y).todense()

    return X,Y

def process_images(names, out_loc, mean=None, sd=None):
    print('Names: ', names)
    dataset = NORBDataset(dataset_root='/dfs/scratch1/thomasat/datasets/norb', names=names)

    Xs = []
    Ys = []

    print('Dataset names: ', dataset.data.keys())

    for name in names:
        X, Y = process_data(dataset.data[name])
        print('X,Y shape: ', X.shape, Y.shape)
        Xs.append(X)
        Ys.append(Y)

    X = np.vstack(Xs)
    Y = np.vstack(Ys)

    if mean is None and sd is None:
        X, mean, sd  = normalize_data(X)
        print('X, Y: ', X.shape, Y.shape)
    else:
        X = apply_normalization(X,mean,sd)

    # Save
    data_dict = {'X': X, 'Y': Y}

    pkl.dump(data_dict, open(out_loc, 'wb'), protocol=2)

    return mean,sd


train_names = ['train' + str(i+1) for i in np.arange(10)]
train_out_loc = '/dfs/scratch1/thomasat/datasets/norb_full/processed_py2_train_' + str(DS_SIZE[0]) + '.pkl'
test_names = ['test' + str(i+1) for i in range(2)]
test_out_loc = '/dfs/scratch1/thomasat/datasets/norb_full/processed_py2_test' + str(DS_SIZE[0]) + '.pkl'

mean, sd = process_images(train_names, train_out_loc)
process_images(test_names, test_out_loc, mean, sd)
