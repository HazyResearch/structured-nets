import sys
import pickle as pkl
sys.path.insert(0, '../../')
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from smallnorb import SmallNORBDataset
from scipy.misc import imresize

MAX_VAL = 255.0
DS_SIZE = (24, 24)
N_CATEGORIES = 5
OUT_LOC = '/dfs/scratch1/thomasat/datasets/smallnorb/processed_py2.pkl'

"""
Downsamples and normalizes.
"""
def process_image(image):
	# Downsample
	ds = imresize(image, DS_SIZE, 'nearest')

	# Normalize
	ds = ds/MAX_VAL

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

dataset = SmallNORBDataset(dataset_root='/dfs/scratch1/thomasat/datasets/smallnorb')

train_X, train_Y = process_data(dataset.data['train'])
test_X, test_Y = process_data(dataset.data['test'])

print('train_X, train_Y, test_X, test_Y: ', train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

# Save
data_dict = {'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y}
pkl.dump(data_dict, open(OUT_LOC, 'wb'), protocol=2)



