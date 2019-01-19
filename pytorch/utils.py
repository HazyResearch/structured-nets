import torch
import torch.nn as nn

def mse_loss(pred, true):
    loss_fn = nn.MSELoss()
    mse = loss_fn(pred, true)
    accuracy = torch.FloatTensor([0])

    return mse, accuracy

def torch_cross_entropy_loss(pred, true):
    # loss_fn = nn.CrossEntropyLoss()
    # _, true_argmax = torch.max(true, 1)
    # cross_entropy = loss_fn(pred, true_argmax)
    cross_entropy = nn.CrossEntropyLoss()(pred, true)

    _, pred_argmax = torch.max(pred, 1)
    correct_prediction = torch.eq(true, pred_argmax)
    accuracy = torch.mean(correct_prediction.float())

    return cross_entropy, accuracy

def cross_entropy_loss(pred, true):
    loss_fn = nn.CrossEntropyLoss()
    _, true_argmax = torch.max(true, 1)
    cross_entropy = loss_fn(pred, true_argmax)

    _, pred_argmax = torch.max(pred, 1)
    correct_prediction = torch.eq(true_argmax, pred_argmax)
    accuracy = torch.mean(correct_prediction.float())

    return cross_entropy, accuracy


def get_commit_id():
  return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])

def descendants(cls):
    """
    Get all subclasses (recursively) of class cls, not including itself
    Assumes no multiple inheritance
    """
    desc = []
    for subcls in cls.__subclasses__():
        desc.append(subcls)
        desc.extend(descendants(subcls))
    return desc


## hack for other code path dependency
def bitreversal_permutation(n):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return perm.squeeze(0)
