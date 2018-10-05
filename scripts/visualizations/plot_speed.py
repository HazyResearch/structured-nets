import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from timeit import default_timer as timer
import timeit
import pickle as pkl
import matplotlib.patches as mpatches
import sys
sys.path.insert(0,'../../pytorch/')

import structure.toeplitz_cpu as toep
import structure.scratch.krylovfast as subd
plt.rcParams['font.family'] = 'serif'

def test_unstructured(n,trials,reps):
    u_setup_str = '''
import numpy as np
np.random.seed(0)
A = np.random.normal(size=({n}, {n}))
v = np.random.normal(size=({n}))
'''.format(n=n)
    return min(timeit.repeat("A @ v", u_setup_str, number = trials, repeat = reps))

def test_toeplitz(n,r,trials,reps):
    t_setup_str = '''
import numpy as np
import structure.toeplitz_cpu as toep
np.random.seed(0)
G = np.random.normal(size=({r}, {n}))
H = np.random.normal(size=({r}, {n}))
v = np.random.normal(size=(1,{n}))
'''.format(n=n,r=r)
    return min(timeit.repeat("toep.toeplitz_mult(G, H, v)", t_setup_str, number = trials, repeat = reps))

def test_lr(n,r,trials,reps):
    lr_setup_str = '''
import numpy as np
np.random.seed(0)
G = np.random.normal(size=({n}, {r}))
H = np.random.normal(size=({r}, {n}))
v = np.random.normal(size={n})
'''.format(n=n,r=r)
    return min(timeit.repeat("Hv = H @ v;G @ Hv", lr_setup_str, number = trials, repeat = reps))


def test_sd(n,r,trials,reps):
    sd_setup_str = '''
import numpy as np
import structure.scratch.krylovfast as subd
np.random.seed(0)
G = np.random.normal(size=({r}, {n}))
H = np.random.normal(size=({r}, {n}))
v = np.random.normal(size=(1,{n}))
K = subd.KrylovMultiply({n}, 1, {r})
KT = subd.KrylovTransposeMultiply({n}, 1, {r})
subd_A = np.random.normal(size=({n}-1))
subd_B = np.random.normal(size=({n}-1))
'''.format(n=n,r=r)
    return min(timeit.repeat("K(subd_A, G, KT(subd_B, H, v))", sd_setup_str, number = trials, repeat = reps))

exps = np.arange(9,16)
sizes = 1 << exps
rs = [1,2,4,8,16]
trials = 1000
reps = 10
out_loc = 'speed_data.p'

data = {}
data['rs'] = rs
data['sizes'] = sizes
data['trials'] = trials
data['reps'] = reps

times_t = np.zeros((len(rs), sizes.size))
times_sd = np.zeros((len(rs), sizes.size))
times_lr = np.zeros((len(rs), sizes.size))
speedups_t = np.zeros((len(rs), sizes.size))
speedups_sd = np.zeros((len(rs), sizes.size))
speedups_lr = np.zeros((len(rs), sizes.size))
unstructured_times = np.zeros(sizes.size)

for idx_n, n in enumerate(sizes):
    unstructured_times[idx_n] = test_unstructured(n,trials,reps)

data['unstructured_times'] = unstructured_times

for idx_r, r in enumerate(rs):
    for idx_n, n in enumerate(sizes):
        t = test_toeplitz(n,r,trials,reps)
        sd = test_sd(n,r,trials,reps)
        lr = test_lr(n,r,trials,reps)
        times_t[idx_r,idx_n] = t
        times_sd[idx_r,idx_n] = sd
        times_lr[idx_r,idx_n] = lr
        data['t'] = times_t
        data['sd'] = times_sd
        data['lr'] = times_lr

pkl.dump(data,open(out_loc,'wb'),protocol=2)
