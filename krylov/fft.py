import numpy as np
import itertools

p=2
d=3
N=p << (d-1)

f = np.arange(N)
print(np.fft.fft(f))


def init(f):
    x   = np.zeros(d*[p], dtype=np.complex_)
    idx = [list(range(p)) for i in range(d)]
    powers  = np.array([p**i for i in range(d)])
    for t in itertools.product(*idx):
        x[t] = f[np.sum(powers*np.array(t))]
    return x
x = init(f)
print(x.shape)

def unshape(x):
    f = np.zeros(p**d, dtype=np.complex_)
    idx = [list(range(p)) for i in range(d)]
    powers  = np.array([p**i for i in range(d)])
    for t in itertools.product(*idx):
        f[np.sum(powers*np.array(t))] = x[t]
    return f


# x = f.reshape([[p]*d]).astype(np.complex_)


# At pass r the layout is
#
# x_0,..., x_{d-r-1}, y_{r-1}, ..., y_{0}
# So at
# r = 0 => x_0,..., x_{d-1}
# r = 1 => x_0,..., x_{d-2}, y_0
# r = 2 => x_0,..., x_{d-3}, y_0, y_1
# r = d => y_0,..., y_{d-1}
#
def pass_it_(x,x_new,r, verbose=False):
    # The index ranges
    # (x_0,...,x_{d-r-2},x_{d-r-1}, y_{0}, .., y_{r-1}, y_r)
    idx     = [list(range(p)) for i in range(d+1)]
    omega   = -2*np.complex(0,1)*np.pi/(p**d)
    powers  = np.array([p**i for i in range(r+1)])
    # powers  = np.array([p**i for i in range(r,-1,-1)])
    for t in itertools.product(*idx):
        # The last index are the ys
        x_base  = t[0:d-r-1]
        x_last  = t[d-r-1] # this is xm
        y_base  = t[d-r:d]
        y_last  = t[d]
        # marginalize out over xm, but keep the ys in the same order?
        new_t  = x_base + y_base    + (y_last,)
        old_t  = x_base + (x_last,) + y_base
        y_sum  = np.sum(np.array(t[d-r:d+1]) * powers) * p**(d-r-1)
        if verbose:
            print(f"x={x_base},{x_last} -> y={y_base},{y_last} :: new={new_t} += old={old_t} y_sum={y_sum} {y_sum*x_last}")
        q      = omega*x_last*y_sum
        x_new[new_t] += x[old_t]*np.exp(q)
    if verbose: print("**")
    return x_new

def pass_it(x,r,verbose=False):
    x_new   = np.zeros(d*[p], dtype=np.complex_)
    return pass_it_(x,x_new,r,verbose=verbose)

def fft_pass(x):
    _x    = np.copy(x)
    x_new = np.zeros(d*[p], dtype=np.complex_)
    for r in range(d):
        pass_it_(_x,x_new,r)
        (_x,x_new) = (x_new,_x)
        x_new[:]  = 0
    return _x

def slow_fft(x):
    y       = np.zeros(x.shape, dtype=np.complex_)
    idx     = [list(range(p)) for i in range(d)]
    omega   = -2*np.complex(0,1)*np.pi/(p**d)
    powers  = np.array([p**i for i in range(d)])
    # powers  = np.array([p**i for i in range(d-1,-1,-1)])
    for t in itertools.product(*idx):
        y_t = np.sum(powers*np.array(t))
        for u in itertools.product(*idx):
            x_t    = np.sum(powers*np.array(u))
            y[t]  += x[u]*np.exp(omega*y_t*x_t)
    return y
