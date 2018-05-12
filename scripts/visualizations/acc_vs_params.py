import matplotlib.pyplot as plt

def update_minmax(mini, maxi, a):
    return min(mini, min(a)), max(maxi, max(a))

def plot_all(ax, n, sd, td, t=None, v=None, h=None, lr=None, u=None, fc=None):
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
    learned_params = [2*n] + list(map(lambda r: 2*n*(r+1), sd['r'])) \
                     + list(map(lambda r: 2*n*(r+3), td['r']))
    learned_acc = [t['acc'][0]] + sd['acc'] + td['acc']
    learned_std = [t['std'][0]] + sd['std'] + td['std']
    minp, maxp = min(learned_params), max(learned_params)
    mina, maxa = min(learned_acc), max(learned_acc)
    ax.errorbar(learned_params, learned_acc, yerr=learned_std, linewidth=2, label='Learned operators (ours)')
    if t is not None:
        t_params = list(map(lambda r: 2*n*r, t['r']))
        minp, maxp = update_minmax(minp, maxp, t_params)
        mina, maxa = update_minmax(mina, maxa, t['acc'])
        ax.errorbar(t_params, t['acc'], yerr=t['std'], linewidth=2, label='Toeplitz-like')
    if v is not None:
        v_params = list(map(lambda r: 2*n*r, v['r'])) # should be +n but looks weird for visualization
        minp, maxp = update_minmax(minp, maxp, v_params)
        mina, maxa = update_minmax(mina, maxa, v['acc'])
        ax.errorbar(v_params, v['acc'], yerr=v['std'], linewidth=2, label='Vandermonde-like')
    if h is not None:
        h_params = list(map(lambda r: 2*n*r, h['r']))
        minp, maxp = update_minmax(minp, maxp, h_params)
        mina, maxa = update_minmax(mina, maxa, h['acc'])
        ax.errorbar(h_params, h['acc'], yerr=h['std'], linewidth=2, label='Hankel-like')
    if lr is not None:
        lr_params = list(map(lambda r: 2*n*r, lr['r']))
        minp, maxp = update_minmax(minp, maxp, lr_params)
        mina, maxa = update_minmax(mina, maxa, lr['acc'])
        ax.errorbar(lr_params, lr['acc'], yerr=lr['std'], linewidth=2, label='Low Rank')
    if u is not None:
        u_params = list(map(lambda h: n*h, u['h']))
        minp, maxp = update_minmax(minp, maxp, u_params)
        mina, maxa = update_minmax(mina, maxa, u['acc'])
        ax.errorbar(u_params, u['acc'], yerr=u['std'], linewidth=2, label='Unconstrained')
    if fc is not None:
        mina, maxa = update_minmax(mina, maxa, [fc[0]])
        ax.plot([minp, maxp], [fc[0], fc[0]], label='Fully Connected', color='black', linewidth=3, linestyle='--')

    ax.set_aspect('auto', adjustable='box')
    ax.set_xlim([minp-1, maxp+1])
    ax.set_ylim([mina-(maxa-mina)*0.1, maxa+(maxa-mina)*0.1])
    # ax.set_xlabel('Total number of parameters')
    # ax.set_ylabel('Test accuracy')
    # if legend == True:
    #     ax.legend(loc='lower right')


fig = plt.figure(figsize=(20,5))

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
    'r': [1, 2, 3, 4],
    'acc': [0.619, 0.69716668, 0.70766664, 0.70833331],
    'std': [0.013044797, 0.019707605, 0.0047667827, 0.005778301]
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
fc = (.6875, 0)
plt.subplot(1,3,1)
plt.title("MNIST noise")
plot_all(fig.axes[0], 784, sd, td, t=t, v=v, h=h, lr=lr, u=None, fc=fc)
fig.axes[0].set_ylabel('Test accuracy')
# plt.show()
# plt.savefig('params_shl_mnist_noise.pdf', bbox_inches='tight')
# plt.clf()

# SHL CIFAR-10 mono
sd = {
    'r': [1],
    'acc': [0.43113336],
    'std': [0.0055289017]
}
td = {
    'r': [1],
    'acc': [0.4564],
    'std': [0.0008751]
}
t = {
    'r': [1,2,3,4],
    'acc': [0.37740001, 0.40759999, 0.41240001, 0.41920003],
'std': [0.0025495042, 0.00050000846, 0.0046396833, 0.0048194025]
    }
v = {
    'r': [1,2,3,4],
    'acc': [0.2764667, 0.31040001, 0.32886663, 0.33543333],
'std': [0.014613534, 0.0050259386, 0.0074763279, 4.7148274e-05]
    }
h = {
    'r': [1,2,3,4],
    'acc': [0.3804667, .39383331, 0.40773332, 0.40799999],
'std': [0.012873066, 0.0071097841, 0.0033767156, 0.0062053697]
    }
lr = {
    'r': [1,2,3,4],
    'acc': [0.18790001, 0.24963333, 0.28833336, 0.31900001],
'std': [0.0012832283, 0.0010208919, 0.00089938374, 0.0048380499]
    }
fc = (0.4708, 0)
# plt.figure()
plt.subplot(1,3,2)
plt.title("CIFAR-10 mono")
plot_all(fig.axes[1], 1024, sd, td, t=t, v=v, h=h, lr=lr, u=None, fc=fc)
fig.axes[1].set_xlabel('Number of Parameters in Hidden Layer')
# fig.axes[1].legend(loc='lower right')
# plt.show()
# plt.savefig('params_shl_cifar_mono.pdf', bbox_inches='tight')
# plt.clf()

# SHL NORB
sd = {
    'r': [1],
    'acc': [0.59593624],
    'std': [0.0068538445]
}
td = {
    'r': [1],
    'acc': [0.59523886],
    'std': [0.0018132643]
}
t = {
    'r': [1,2,3,4],
    'acc': [0.48894033, 0.54002631, 0.55142885, 0.56502056],
    'std': [0.0041263779, 0.0099200783, 0.0024635822, 0.0061167572]
}
v = {
    'r': [1,2,3,4],
    'acc': [0.3664552, 0.42934385, 0.4352195, 0.44603908],
    'std': [0.0032585002, 0.0087335464, 0.0061135041, 0.0024887063]
}
h = {
    'r': [1,2,3,4],
    'acc': [0.47990969, 0.53255033, 0.5393576, 0.53800869],
    'std': [0.0096045565, 0.004828494, 0.0089905132, 0.0049836915]
}
lr = {
    'r': [1,2,3,4],
    'acc': [0.3285208, 0.37442842, 0.39659354, 0.43265319],
    'std': [0.0026101458, 0.0013176826, 0.0001270434, 0.001628752]
}
fc = (0.6041038, 0)
# plt.figure()
plt.subplot(1,3,3)
plt.title("NORB")
plot_all(fig.axes[2], 784, sd, td, t=t, v=v, h=h, lr=lr, u=None, fc=fc)
# fig.axes[2].legend(loc='lower right')
# plt.show()
# plt.savefig('params_shl_norb.pdf', bbox_inches='tight')
# plt.clf()


# plt.show()
plt.savefig('acc_vs_params_shl.pdf', bbox_inches='tight')
plt.clf()

# CNN last layer
fig = plt.figure(figsize=(20,5))


# CNN MNIST-noise
sd = {
    'r': [1],
    'acc': [0.9265],
    'std': [0.0047081495]
}
td = {
    'r': [1],
    'acc': [0.93533343],
    'std': [0.0051044598]
}
t = {
    'r': [1,2,3,4],
    'acc': [0.90533328, 0.90933341, 0.90350002, 0.91433334],
    'std': [0.0058642207, 0.0024944325, 0.0056124986, 0.0067618182]
}
v = {
    'r': [1,2,3,4],
    'acc': [0.76366663, 0.861, 0.85799998, 0.84683329],
    'std': [0.055379499, 0.0024832655, 0.007560859, 0.011813368]
}
h = {
    'r': [1,2,3,4],
    'acc': [0.90700006, 0.90633339, 0.90249997, 0.91150004],
    'std': [0.001080128, 0.0031710495, 0.0051153307, 0.0040207645]
}
lr = {
    'r': [1,2,3,4],
    'acc': [0.39333335, 0.66433334, 0.80366665, 0.84183329],
    'std': [0.02600107, 0.023686616, 0.014401001, 0.010217078]
}
fc = (0.903, 0)
plt.subplot(1,3,1)
plt.title("MNIST noise")
plot_all(fig.axes[0], 784, sd, td, t=t, v=v, h=h, lr=lr, u=None, fc=fc)
fig.axes[0].set_ylabel('Test accuracy')

# CNN CIFAR-10 mono
sd = {
    'r': [1],
    'acc': [0.64996666],
    'std': [0.0088710589]
}
td = {
    'r': [1],
    'acc': [0.66089994],
    'std': [0.0076345755]
}
t = {
    'r': [1,2,3,4],
    'acc': [0.64956665, 0.65023333, 0.65143329, 0.64866662],
    'std': [0.00069441635, 0.010134532, 0.0073803198, 0.0027475108]
}
v = {
    'r': [1,2,3,4],
    'acc': [0.49699998, 0.56516665, 0.58923334, 0.59],
    'std': [0.012887465, 0.010227513, 0.0064116176, 0.01]
}
h = {
    'r': [1,2,3,4],
    'acc': [0.64123327, 0.64403337, 0.6487667, 0.64303333],
    'std': [0.0028755669, 0.0066979155, 0.0085495887, 0.0075429156]
}
lr = {
    'r': [1,2,3,4],
    'acc': [0.28889999, 0.45389998, 0.53403336, 0.57600003],
    'std': [0.0041817101, 0.0045460523, 0.004343845, 0.0048006908]
}
fc = (0.6528, 0)
plt.subplot(1,3,2)
plt.title("CIFAR-10 mono")
plot_all(fig.axes[1], 1024, sd, td, t=t, v=v, h=h, lr=lr, u=None, fc=fc)
fig.axes[1].set_xlabel('Number of Parameters in CNN Last Layer')
# fig.axes[1].legend(loc='lower right')


# CNN NORB
sd = {
    'r': [1],
    'acc': [0.69955987],
    'std': [0.0057070036]
}
td = {
    'r': [1],
    'acc': [0.70161748],
    'std': [0.0034562966]
}
t = {
    'r': [1,2,3,4],
    'acc': [0.69, 0.697308, 0.697, 0.69480449],
    'std': [0.005, 0.0055511161, 0.005, 0.0074541033]
}
v = {
    'r': [1,2,3,4],
    'acc': [0.59256977, 0.67032462, 0.67778349, 0.66851282],
    'std': [0.044178825, 0.010677353, 0.0046389499, 0.0086385822]
}
h = {
    'r': [1,2,3,4],
    'acc': [0.67735481, 0.6791895, 0.6919753, 0.69066644],
    'std': [0.0094787478, 0.0045856023, 0.0037513557, 0.0023269763]
}
lr = {
    'r': [1,2,3,4],
    'acc': [0.47555444, 0.62934959, 0.66097963, 0.66602081],
    'std': [0.0069264239, 0.0041847723, 0.0047160424, 0.0028731704]
}
fc = (0.73229307, 0)
plt.subplot(1,3,3)
plt.title("NORB")
plot_all(fig.axes[2], 784, sd, td, t=t, v=v, h=h, lr=lr, u=None, fc=fc)
fig.axes[2].legend(loc='lower right')


plt.savefig('acc_vs_params_cnn.pdf', bbox_inches='tight')
