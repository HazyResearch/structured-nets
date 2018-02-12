import pickle
import numpy as np
import matplotlib.pyplot as plt

arrays = pickle.load(open('mnist_noise_toep_dr2_0.pkl', 'rb'), encoding='bytes')
G_toeplitz, H_toeplitz, W_toeplitz = [arrays[key] for key in [b'G', b'H', b'W']]
arrays = pickle.load(open('mnist_noise_circ_0.pkl', 'rb'), encoding='bytes')
A_subdiag, B_subdiag, G_subdiag, H_subdiag, W_subdiag = [arrays[key] for key in [b'A', b'B', b'G', b'H', b'W']]
arrays = pickle.load(open('mnist_noise_trid_2.pkl', 'rb'), encoding='bytes')
A_tridiag, B_tridiag, G_tridiag, H_tridiag, W_tridiag = [arrays[key] for key in [b'A', b'B', b'G', b'H', b'W']]
arrays = pickle.load(open('mnist_noise_unconstr_0.pkl', 'rb'), encoding='bytes')
W_unconstrained = arrays[b'W']

plt.figure(figsize=(5, 4.75))
plt.stem(abs(np.fft.fft(np.diag(A_subdiag, -1) - 1))[:100])
plt.xlabel('Frequency', fontsize=16)
plt.ylabel('Magnitude', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(f'../paper/figs/frequency_subdiag_subdiag.pdf', bbox_inches='tight')
plt.close()

plt.figure()
plt.stem(abs(np.fft.fft(np.diag(A_tridiag, -1) - 1))[:100])
plt.xlabel('Frequency', fontsize=16)
plt.ylabel('Magnitude', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(f'../paper/figs/frequency_subdiag_tridiag.pdf', bbox_inches='tight')
plt.close()

plt.figure()
plt.matshow(np.log(1 + np.abs(W_toeplitz.T)), cmap='hot', interpolation='none')
plt.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
plt.savefig(f'../paper/figs/heatmap_weight_toeplitz.pdf', dpi=600, bbox_inches='tight')
plt.close()
plt.figure()
plt.matshow(np.log(1 + np.abs(W_subdiag.T)), cmap='hot', interpolation='none')
plt.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
plt.savefig(f'../paper/figs/heatmap_weight_subdiag.pdf', dpi=600, bbox_inches='tight')
plt.close()

corner_A = A[0, -1]
t = np.hstack([subdiag_A, np.array(corner_A)])
plt.imshow(t.reshape(28, -1) - 1, cmap='gray')
plt.clf()
plt.plot(subdiag_A)
T = A - np.diag(subdiag_A, -1)

plt.matshow(B)
subdiag_B  = np.diag(B, -1)
corner_B = B[0, -1]
t = np.hstack([subdiag_B, -np.array(corner_B)])
plt.imshow(t.reshape(28, -1), cmap='gray')
plt.clf()
plt.plot(subdiag_B)

u, s, v = np.linalg.svd(W.T)
plt.clf()
plt.figure()
plt.plot(s)
plt.clf()
plt.plot(v[0])

num = 10
plt.figure(figsize=(5 * num, 5))
for i in range(num):
    ax = plt.subplot(1, num, i+1)
    t = v[i].reshape((28, -1))
    ax.imshow(t, cmap='gray')
    ax.axis('off')

plt.close()

# plt.matshow(W, cmap='gray')
plt.matshow(W, cmap='gray')
plt.matshow(np.log(1 + np.abs(W_toeplitz)), cmap='hot')
plt.matshow(np.log(1 + np.abs(W_subdiag)), cmap='hot')
f = np.fft.fft2(W)
plt.matshow(abs(f), cmap='gray')
plt.matshow(np.log(1 + abs(f)), cmap='gray')
plt.matshow(fftpack.dct(W), cmap='gray')
plt.matshow(np.log(1 + abs(fftpack.dct(W))), cmap='gray')

from scipy.linalg import circulant
from scipy import fftpack
c = circulant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.matshow(c)
c_f = np.fft.fft2(c)
c_c = fftpack.dct(c)
plt.matshow(abs(c_c))
