import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

n_iters = 1000
step = 100
n = 50
mom = 0.9
prefix = '../../results/mom'

xs = np.arange(0, n_iters, step)
trid_corner = pkl.load(open(prefix + str(mom) + '_' + 'toeplitz_tridiagonal_corner_losses_' + str(n) + '.p', 'rb'))
circ = pkl.load(open(prefix + str(mom) + '_' + 'toeplitz_circulant_sparsity_losses_' + str(n) + '.p', 'rb'))
unconstr = pkl.load(open(prefix + str(mom) + '_' + 'toeplitz_unconstrained_losses_' + str(n) + '.p', 'rb'))
toep = pkl.load(open(prefix + str(mom) + '_' + 'toeplitz_toeplitz_like_losses_' + str(n) + '.p', 'rb'))
hank = pkl.load(open(prefix + str(mom) + '_' + 'toeplitz_hankel_like_losses_' + str(n) + '.p', 'rb'))
van = pkl.load(open(prefix + str(mom) + '_' + 'toeplitz_vandermonde_like_losses_' + str(n) + '.p', 'rb'))

plt.semilogy(xs, unconstr, label='Unconstrained', marker='o')
plt.semilogy(xs, trid_corner, label='Learned tridiagonal + corner operators, r=2', linestyle='--', linewidth=10)
plt.semilogy(xs, toep, label='Toeplitz-like, r=2', linewidth=3)
plt.semilogy(xs, hank, label='Hankel-like, r=2', linewidth=3)
plt.semilogy(xs, van, label='Vandermonde-like, r=2', linewidth=3)
plt.semilogy(xs, circ, label='Learned circulant sparsity pattern operators, r=2', linestyle='-', linewidth=5)

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Toeplitz, mom=' + str(mom) + ', n=' + str(n))

plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), fontsize=10)

plt.savefig("toeplitz_operators_n" + str(n) + 'mom_' + str(mom) + ".png", bbox_inches="tight")

plt.clf()