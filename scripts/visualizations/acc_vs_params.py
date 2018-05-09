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

# SHL MNIST noise
sd = {
    'r': [1],
    'acc': [0.780],
    'std': [0.012456593]
}
td = {
    'r': [1],
    'acc': [0.79433328],
    'std': [0.0087971017]
}
t = {
    'r': [2, 4],
    'acc': [0.69716668, 0.70833331],
    'std': [0.019707605, 0.005778301]
}
v = {
    'r': [1,2,3,4],
    'acc': [0.45233333, 0.56750005, 0.60416669, 0.61383337],
    'std': [0.036148615, 0.018170487, 0.0040276861, 0.0034237648]
}
h = {
    'r': [1,2,3,4],
    'acc': [0.69199997, 0.71716666, 0.71749997, 0.71516663],
    'std': [0.0051153307, 0.02878754, 0.018828174, 0.027265172]
}
lr = {
    'r': [1,2,3,4],
    'acc': [0.24066667, 0.37966666, 0.44949999, 0.50933331],
    'std': [0.016624946, 0.002953345, 0.0028577377, 0.0042491788]
}
fc = (.68466663, 0)
plt.figure()
plot_all(784, sd, td, t=t, v=v, h=h, lr=None, u=None, fc=fc)
plt.show()

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
