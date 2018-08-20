# Download from http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007

import numpy as np
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
from data_utils import normalize_data, apply_normalization

def process_data(data):
    X = data[:, :-1]
    Y = np.expand_dims(data[:, -1], 1)

    # Y must be one-hot
    enc = OneHotEncoder()
    Y = enc.fit_transform(Y).todense()
    return X,Y

train_loc = '/dfs/scratch1/thomasat/datasets/mnist_bg_rot/mnist_all_background_images_rotation_normalized_train_valid.amat'
test_loc = '/dfs/scratch1/thomasat/datasets/mnist_bg_rot/mnist_all_background_images_rotation_normalized_test.amat'
train_out = '/dfs/scratch1/thomasat/datasets/mnist_bg_rot/train_normalized'
test_out = '/dfs/scratch1/thomasat/datasets/mnist_bg_rot/test_normalized'

train_data = np.genfromtxt(train_loc)
train_X, train_Y = process_data(train_data)

test_data = np.genfromtxt(test_loc)
test_X, test_Y = process_data(test_data)

# Normalize
train_X, mean, sd = normalize_data(train_X)
test_X = apply_normalization(test_X, mean, sd)

# Save
print('test_X, test_Y shape: ', test_X.shape, test_Y.shape)
print('train_X, train_Y shape: ', train_X.shape, train_Y.shape)
train = {'X': train_X, 'Y': train_Y}
test = {'X': test_X, 'Y': test_Y}

pkl.dump(train, open(train_out, 'wb'), protocol=2)
pkl.dump(test, open(test_out, 'wb'), protocol=2)
print('Saved train to: ', train_out)
print('Saved test to: ', test_out)
