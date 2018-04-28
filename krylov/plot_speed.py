import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import toeplitz_cpu as toep
import triXXF as subd


# sizes = [1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15]
exps = np.arange(7, 12)
sizes = 1 << exps

ranks = [1,4]

fc_times = np.zeros(exps.size)
t_times = np.zeros((len(ranks), exps.size))
sd_times = np.zeros((len(ranks), exps.size))
# for i, n in enumerate(sizes):
trials = 100
for idx_n, n in enumerate(sizes):
    v = np.random.random(n)

    # test unconstrained
    A = np.random.random((n, n))

    start = timer()
    A @ v
    end = timer()
    # print(f"Elapsed time {end-start}")
    fc_times[idx_n] = (end-start)

    for idx_r, r in enumerate(ranks):
        v = np.random.normal(scale=0.01, size=(1, n))
        G = np.random.normal(scale=0.01, size=(r, n))
        H = np.random.normal(scale=0.01, size=(r, n))

        # test toeplitz_like
        start = timer()
        [toep.toeplitz_mult(G, H, v) for _ in range(trials)]
        end = timer()
        t_times[idx_r, idx_n] = (end-start)/trials

        # test subdiagonal
        subd_A = np.random.normal(scale=0.01, size=(n-1))
        subd_B = np.random.normal(scale=0.01, size=(n-1))
        K = subd.KrylovMultiply(n, 1, r)
        KT = subd.KrylovTransposeMultiply(n, 1, r)
        start = timer()
        [K(subd_A, G, KT(subd_B, H, v)) for _ in range(trials)]
        end = timer()
        sd_times[idx_r, idx_n] = (end-start)/trials
print(fc_times)
print(t_times)
print(sd_times)

plt.figure()
plt.semilogy(sizes, fc_times, label='Fully Connected')
for i, r in enumerate(ranks):
    plt.semilogy(sizes, t_times[i], label='Toeplitz-like r'+str(r))
    plt.semilogy(sizes, sd_times[i], label='Scale-cycle r'+str(r))
plt.xscale('log', basex=2)
plt.xlabel("Dimension")
plt.ylabel("Computation Cost")
plt.legend()
# plt.show()
plt.savefig('speed.pdf')
