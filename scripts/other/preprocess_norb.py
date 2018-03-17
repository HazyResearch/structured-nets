import sys
import pickle as pkl
sys.path.insert(0, '../../')
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from norb import NORBDataset
from scipy.misc import imresize

MAX_VAL = 255.0
DS_SIZE = (27, 27)
N_CATEGORIES = 6
TRAIN = True#False

names = ['train' + str(i+1) for i in range(10)]
OUT_LOC = '/dfs/scratch1/thomasat/datasets/norb/processed_py2_train.pkl'
if not TRAIN:
	names = ['test' + str(i+1) for i in range(2)]
	OUT_LOC = '/dfs/scratch1/thomasat/datasets/norb/processed_py2_test.pkl'


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
		#print('this_category: ', this_category)
		X.append(process_image(this_image))
		Y.append(this_category)

	X = np.array(X)
	Y = np.array(Y)
	Y = np.expand_dims(Y, 1)
	enc = OneHotEncoder(N_CATEGORIES)
	Y = enc.fit_transform(Y).todense()

	return X,Y

print('Names: ', names)
dataset = NORBDataset(dataset_root='/dfs/scratch1/thomasat/datasets/norb')

Xs = []
Ys = []


for name in names:
	X, Y = process_data(dataset.data[name])
	print('X,Y shape: ', X.shape, Y.shape)
	Xs.append(X)
	Ys.append(Y)

X = np.vstack(Xs)
Y = np.vstack(Ys)


print('X, Y: ', X.shape, Y.shape)


# Save
data_dict = {'X': X, 'Y': Y}

pkl.dump(data_dict, open(OUT_LOC, 'wb'), protocol=2)

