import numpy as np
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
from data_utils import normalize_data, apply_normalization

# Download from http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007

n_variations = 6
for idx in np.arange(1, n_variations+1):
    data_loc = '/dfs/scratch1/thomasat/datasets/mnist_noise/mnist_noise_variations_all_' + str(idx) + '.amat'
    train_out = '/dfs/scratch1/thomasat/datasets/mnist_noise/train_' + str(idx)
    test_out = '/dfs/scratch1/thomasat/datasets/mnist_noise/test_' + str(idx)
    test_size = 2000 # As specified in http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007#Downloadable_datasets

    data = np.genfromtxt(data_loc)
    X = data[:, :-1]
    Y = np.expand_dims(data[:, -1], 1)

    # Y must be one-hot
    enc = OneHotEncoder()
    Y = enc.fit_transform(Y).todense()
    # Split into train, val, test
    # Shuffle the data
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)

    train_idx = idx[:-test_size]
    test_idx = idx[-test_size:]

    assert train_idx.size == (X.shape[0] - test_size)
    assert test_idx.size == test_size

    test_X = X[test_idx, :]
    test_Y = Y[test_idx, :]
    train_X = X[train_idx, :]
    train_Y = Y[train_idx, :]

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
