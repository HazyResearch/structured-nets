import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

idx = 1
data_loc = '/dfs/scratch1/thomasat/datasets/mnist_noise/mnist_noise_variations_all_' + str(idx) + '.amat'

np.random.seed(0)
data = np.genfromtxt(data_loc)
samples = np.random.choice(data.shape[0], 4)
X = data[samples, :-1].reshape((-1, 28, 28))

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
plt.savefig('mnist_noise_digits.png', bbox_inches='tight')
plt.close()
