import numpy as np
import pickle as pkl
import os
from sklearn.preprocessing import OneHotEncoder                                                                
import matplotlib.pyplot as plt

# Assumes 3 input channels
# Converts to grayscale and normalizes images
def convert_grayscale_and_normalize(data, img_size=32, max_val=255.0):
    n = data.shape[0]
    channel_size = int(data.shape[1]/3)
    print('channel_size:', channel_size)
    im_r = data[:, 0:channel_size].reshape((n, img_size, img_size))             
    im_g = data[:, channel_size:2*channel_size].reshape((n, img_size, img_size))
    im_b = data[:, 2*channel_size:].reshape((n, img_size, img_size))
    img = np.stack((im_r, im_g, im_b), axis=-1)
    avg_img = np.mean(img, axis=-1)
    data = avg_img.reshape((n, img_size*img_size))
    return data/max_val

def load_and_preprocess_data(loc):
    data_dict = pkl.load(open(loc, 'rb'),encoding='latin1')
    X = data_dict['data']
    if grayscale:
        X = convert_grayscale_and_normalize(X)
    Y = np.array(data_dict['labels'])
    Y = np.expand_dims(Y,1)
    Y = enc.fit_transform(Y).todense()

    print('X.shape, Y.shape: ', X.shape, Y.shape)
    return X,Y

grayscale = True
train_batches = 5
data_dir = '/dfs/scratch1/thomasat/datasets/cifar10'
test_loc = os.path.join(data_dir,'test_batch')
train_out = '/dfs/scratch1/thomasat/datasets/cifar10_combined/train'
test_out = '/dfs/scratch1/thomasat/datasets/cifar10_combined/test'
if grayscale:
    train_out += '_grayscale'
    test_out += '_grayscale'

# Prepare training data
train_X = []
train_Y = []

enc = OneHotEncoder()

for i in range(train_batches):
    this_batch_loc = os.path.join(data_dir, 'data_batch_' + str(i+1)) 
    X,Y = load_and_preprocess_data(this_batch_loc)
    train_X.append(X)
    train_Y.append(Y)

# Concatenate
train_X = np.vstack(train_X)
train_Y = np.vstack(train_Y)


# Shuffle
idx = np.arange(0, train_X.shape[0])  
np.random.shuffle(idx)
train_X = train_X[idx,:]
train_Y = train_Y[idx,:]

print('train_X.shape, train_Y.shape: ', train_X.shape, train_Y.shape)

# Prepare test data
test_X,test_Y = load_and_preprocess_data(test_loc)
print('test_X.shape, test_Y.shape: ', test_X.shape, test_Y.shape)

# Save
train = {'X': train_X, 'Y': train_Y}
test = {'X': test_X, 'Y': test_Y}

pkl.dump(train, open(train_out, 'wb'), protocol=2)
pkl.dump(test, open(test_out, 'wb'), protocol=2)
print('Saved train to: ', train_out)
print('Saved test to: ', test_out)
