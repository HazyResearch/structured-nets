import matplotlib.pyplot as plt

def update_minmax(mini, maxi, a):
    return min(mini, min(a)), max(maxi, max(a))

def plot_all(n, sd, td, t=None, v=None, h=None, lr=None, u=None, fc=None):
    """
    pass in as dict of three arrays: r, acc, std
    sd: subdiagonal
    td: tridiagonal
    t:  toeplitz
    v:  vandermonde
    h:  hankel
    lr: low rank

    pass in as dict of three arrays: h (hidden units), acc, std
    u:  unconstrained

    pass in as tuple of two numbers: acc, std
    fc: n hidden units, fully connected
    """
    learned_params = list(map(lambda r: 2*n*(r+1), sd['r'])) \
                     + list(map(lambda r: 2*n*(r+3), td['r']))
    learned_acc = sd['acc'] + td['acc']
    learned_std = sd['std'] + td['std']
    minp, maxp = min(learned_params), max(learned_params)
    mina, maxa = min(learned_acc), max(learned_acc)
    plt.errorbar(learned_params, learned_acc, yerr=learned_std, label='Learned operators (ours)')
    if t is not None:
        t_params = list(map(lambda r: 2*n*r, t['r']))
        minp, maxp = update_minmax(minp, maxp, t_params)
        mina, maxa = update_minmax(mina, maxa, t['acc'])
        plt.errorbar(t_params, t['acc'], yerr=t['std'], label='Toeplitz-like')
    if v is not None:
        v_params = list(map(lambda r: 2*n*r+n, v['r']))
        minp, maxp = update_minmax(minp, maxp, v_params)
        mina, maxa = update_minmax(mina, maxa, v['acc'])
        plt.errorbar(v_params, v['acc'], yerr=v['std'], label='Vandermonde-like')
    if h is not None:
        h_params = list(map(lambda r: 2*n*r, h['r']))
        minp, maxp = update_minmax(minp, maxp, h_params)
        mina, maxa = update_minmax(mina, maxa, h['acc'])
        plt.errorbar(h_params, h['acc'], yerr=h['std'], label='Hankel-like')
    if lr is not None:
        lr_params = list(map(lambda r: 2*n*r, lr['r']))
        minp, maxp = update_minmax(minp, maxp, lr_params)
        mina, maxa = update_minmax(mina, maxa, lr['acc'])
        plt.errorbar(lr_params, lr['acc'], yerr=lr['std'], label='Low Rank')
    if u is not None:
        u_params = list(map(lambda h: n*h, u['h']))
        minp, maxp = update_minmax(minp, maxp, u_params)
        mina, maxa = update_minmax(mina, maxa, u['acc'])
        plt.errorbar(u_params, u['acc'], yerr=u['std'], label='Unconstrained')
    if fc is not None:
        mina, maxa = update_minmax(mina, maxa, [fc[0]])
        plt.plot([minp, maxp], [fc[0], fc[0]], label='Fully Connected', color='black', linewidth=3, linestyle='--')

    plt.xlim([minp-1, maxp+1])
    plt.ylim([mina-(maxa-mina)*0.1, maxa+(maxa-mina)*0.1])
    plt.xlabel('Total number of parameters')
    plt.ylabel('Test accuracy')
    plt.legend(loc='lower right')


# CNN MNIST noise
sd = {
    'r': [1],
    'acc': [0.932],
    'std': [0]
}
td = {
    'r': [1],
    'acc': [0.938],
    'std': [0]
}
t = {
    'r': [2, 4],
    'acc': [0.912, 0.923],
    'std': [0, 0]
}
v = {}
h = {}
lr = {}
fc = (0.905, 0)
plt.figure()
plot_all(784, sd, td, t=t, v=None, h=None, lr=None, u=None, fc=fc)
plt.show()

# CNN CIFAR-10 mono
sd = {
    'r': [1],
    'acc': [0.659],
    'std': [0]
}
td = {
    'r': [1],
    'acc': [0.672],
    'std': [0]
}
t = {
    'r': [2, 4],
    'acc': [0.663, 0.652],
    'std': [0, 0]
}
v = {}
h = {}
lr = {}
fc = (0.665, 0)
plt.figure()
plot_all(1024, sd, td, t=t, v=None, h=None, lr=None, u=None, fc=fc)
plt.show()
