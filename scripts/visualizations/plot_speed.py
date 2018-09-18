import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from timeit import default_timer as timer

import sys
sys.path.insert(0,'../../pytorch/')

import structure.toeplitz_cpu as toep
import structure.scratch.krylovfast as subd

# sizes = [1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15]
exps = np.arange(7,16)#np.arange(7, 16)
sizes = 1 << exps

ranks = [1]

# For each trial, randomly generate A and v for unconstrained
# For structured classes, randomly generate v,G,H,subd_A,subd_B in each trial
# Report mean and standard deviation

fc_times = [] # n x num_trials
t_times = {}#r -> n x num_trials
sd_times = {}#r -> n x num_trials
# for i, n in enumerate(sizes):

trials = 100
for idx_n, n in enumerate(sizes):
    fc_times.append([])
    print('FC, N: ', n)

    # test unconstrained
    for _ in range(trials):
        v = np.random.normal(scale=0.01, size=(n))
        A = np.random.normal(scale=0.01, size=(n, n))
        start = timer()
        A @ v
        end = timer()
        fc_times[idx_n].append(end-start)
    #print('N, fc_time: ', fc_times[idx_n])

fc_times = np.array(fc_times)

for idx_r, r in enumerate(ranks):
    t_times[idx_r] = []
    sd_times[idx_r] = []
    for idx_n, n in enumerate(sizes):
        print('T/SD, N: ', n)
        # test toeplitz_like
        t_times[idx_r].append([])
        sd_times[idx_r].append([])
        for _ in range(trials):
            v = np.random.normal(scale=0.01, size=(1, n))
            G = np.random.normal(scale=0.01, size=(r, n))
            H = np.random.normal(scale=0.01, size=(r, n))
            start = timer()
            toep.toeplitz_mult(G, H, v)
            end = timer()
            t_times[idx_r][idx_n].append(end-start)

        # test subdiagonal
        for _ in range(trials):
            v = np.random.normal(scale=0.01, size=(1, n))
            G = np.random.normal(scale=0.01, size=(r, n))
            H = np.random.normal(scale=0.01, size=(r, n))
            subd_A = np.random.normal(scale=0.01, size=(n-1))
            subd_B = np.random.normal(scale=0.01, size=(n-1))
            K = subd.KrylovMultiply(n, 1, r)
            KT = subd.KrylovTransposeMultiply(n, 1, r)
            start = timer()
            K(subd_A, G, KT(subd_B, H, v))
            end = timer()
            sd_times[idx_r][idx_n].append(end-start)

    t_times[idx_r] = np.array(t_times[idx_r])
    sd_times[idx_r] = np.array(sd_times[idx_r])

print('FC: ', fc_times.shape)
print('T: ', t_times[0].shape)
print('SD: ', sd_times[0].shape)


# Compute mean and stdev

plt.figure()
plt.errorbar(sizes, np.mean(fc_times,axis=1), yerr=np.std(fc_times,axis=1), label='Fully Connected')
for i, r in enumerate(ranks):
    plt.errorbar(sizes, np.mean(t_times[i],axis=1), yerr=np.std(t_times[i],axis=1),label='Toeplitz-like r'+str(r))
    plt.errorbar(sizes, np.mean(sd_times[i],axis=1), yerr=np.std(sd_times[i],axis=1),label='Scale-cycle r'+str(r))
plt.xscale('log', basex=2)
plt.yscale('log')
plt.xlabel("Dimension")
plt.ylabel("Computation Cost")
plt.legend()
# plt.show()
plt.savefig('speed.pdf')
