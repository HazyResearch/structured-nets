import matplotlib.pyplot as plt

# For MNIST noise
fixed_xs = [10202, 11770, 13338, 14906]
lowrank = [0.2403, 0.377, 0.4577, 0.546]
lowrank_std = [0.0184407, 0.004, 0.00334066, 0.006]
toep = [0.62525, 0.681, 0.6758, 0.712]
toep_std = [0.00125, 0.017, 0.0227389, 0.012]
hank = [0.66175, 0.696667, 0.70475, 0.704]
hank_std = [0.0284044, 0.016705, 0.0174696, 0.0194551]
van = [0.434, 0.5667, 0.584125, 0.626]
van_std = [0.00147196, 0.00804115, 0.00867016, 0.005]

# 0.6761 0.002245
unconstr_val = 0.6761
unconstr_h784 = [unconstr_val for x in range(len(fixed_xs))]
unconstr_h784std = [0,0,0,0]

unconstr = [0.5941, 0.6135, 0.6064, 0.616]
unconstr_std = [0.0118929, 0.00749667, 0.0073512, 0.0049699]

learned_xs = [10202, 10454, 11454, 11770, 14904]
learned = [toep[0], 0.6807, 0.7447, 0.765, 0.784]
learned_std = [0.0, 0.0142253, 0.0216809, 0.018, 0.018]

plt.errorbar(learned_xs, learned, yerr=learned_std, label='Learned operators (ours)')
plt.errorbar(fixed_xs, hank, yerr=hank_std, label='Hankel-like')
plt.errorbar(fixed_xs, toep, yerr=toep_std, label='Toeplitz-like')
plt.errorbar(fixed_xs, van, yerr=van_std, label='Vandermonde-like')
plt.errorbar(fixed_xs, lowrank, yerr=lowrank_std, label='Low rank')
plt.errorbar(fixed_xs, unconstr_h784, yerr=unconstr_h784std, label='Unconstrained, ' + r'$h=784$', color='black', linewidth=5, linestyle='--')
plt.errorbar(fixed_xs, unconstr, yerr=unconstr_std, label='Unconstrained')

plt.xlim([10202, 14906])
plt.ylim([0.2, 0.8])
plt.xlabel('Total number of parameters')
plt.ylabel('Test accuracy')
plt.legend(loc='lower right')
plt.savefig("mnist_noise_params.png", bbox_inches="tight")
plt.clf()

# For CIFAR-10 grayscale
lowrank = [0.18576, 0.24828 , 0.2868 , 0.31054]
lowrank_std = [ 0.000682932, 0.00265887, 0.00272177, 0.00160325]
toep = [0.32765, 0.33692, 0.342133, 0.3472]
toep_std = [0.0161717, 0.00563894, 0.00694087, 0.0106909]
hank = [0.3202, 0.329333, 0.3317, 0.34005]
hank_std = [0.00645032, 0.00373839, 0.00659999, 0.00155]
van = [0.24382, 0.26516, 0.29505, 0.31405]
van_std = [0.00246284,  0.00698873, 0.00635, 0.00225]
learned = [toep[0], 0.36406, 0.3966]
learned_std = [0.0, 0.0104458, 0.000300005]
fixed_xs = [13322, 15370, 17418, 19466]
learned_xs = [13322, 15370, 19466]

unconstr_val = 0.4326
unconstr_h784 = [unconstr_val for x in range(len(fixed_xs))]
unconstr = [0.33606, 0.34328,0.35444, 0.36318]
unconstr_std = [0.00389287, 0.0071042, 0.00625223, 0.00380337]


plt.errorbar(learned_xs, learned, yerr=learned_std, label='Learned operators (ours)')
plt.errorbar(fixed_xs, hank, yerr=hank_std, label='Hankel-like')
plt.errorbar(fixed_xs, toep, yerr=toep_std, label='Toeplitz-like')
plt.errorbar(fixed_xs, van, yerr=van_std, label='Vandermonde-like')
plt.errorbar(fixed_xs, lowrank, yerr=lowrank_std, label='Low rank')
plt.errorbar(fixed_xs, unconstr_h784, yerr=unconstr_h784std, label='Unconstrained, ' + r'$h=104$', color='black', linewidth=5, linestyle='--')
plt.errorbar(fixed_xs, unconstr, yerr=unconstr_std, label='Unconstrained')

plt.xlim([13322, 19466])
plt.xlabel('Total number of parameters')
plt.ylabel('Test accuracy')
plt.legend(loc='lower right')
#plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), fontsize=10)
plt.savefig("cifar10_params.png", bbox_inches="tight")
plt.clf()

