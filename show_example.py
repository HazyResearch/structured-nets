import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle as pkl

# MNIST variants
# idx = 1
# # data_loc = '/dfs/scratch1/thomasat/datasets/mnist_noise/mnist_noise_variations_all_' + str(idx) + '.amat'
# data_loc = '/dfs/scratch1/thomasat/datasets/mnist_bg_rot/mnist_all_background_images_rotation_normalized_train_valid.amat'
# data = np.genfromtxt(data_loc)
# save_loc = 'mnist_bg_rot_digits.png'

# np.random.seed(1)
# samples = np.random.choice(data.shape[0], 4)
# X = data[samples, :-1].reshape((-1, 28, 28))


# NORB
data_loc = '/dfs/scratch1/thomasat/datasets/norb_full/processed_py2_train_28.pkl'
data = pkl.load(open(data_loc, 'rb'))
data = data['X']

save_loc = 'norb_digits.png'
# self.train_Y = data['Y']

np.random.seed(1)
samples = np.random.choice(data.shape[0], 4)
X = data[samples, :].reshape((-1, 28, 28))

fig = plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.axis('off')
plt.imshow(X[0,:], cmap='gray', interpolation='nearest')
plt.subplot(2,2,2)
plt.axis('off')
plt.imshow(X[1,:], cmap='gray', interpolation='nearest')
plt.subplot(2,2,3)
plt.axis('off')
plt.imshow(X[2,:], cmap='gray', interpolation='nearest')
plt.subplot(2,2,4)
plt.axis('off')
plt.imshow(X[3,:], cmap='gray', interpolation='nearest')
fig.subplots_adjust(wspace=0,hspace=0)
plt.savefig(save_loc, bbox_inches='tight')
plt.close()




