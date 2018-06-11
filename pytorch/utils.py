import torch
import torch.nn as nn

def mse_loss(pred, true):
    loss_fn = nn.MSELoss()
    mse = loss_fn(pred, true)
    accuracy = torch.FloatTensor([0])

    return mse, accuracy

def cross_entropy_loss(pred, true):
    loss_fn = nn.CrossEntropyLoss()
    _, true_argmax = torch.max(true, 1)
    cross_entropy = loss_fn(pred, true_argmax)

    _, pred_argmax = torch.max(pred, 1)
    correct_prediction = torch.eq(true_argmax, pred_argmax)
    accuracy = torch.mean(correct_prediction.float())

    return cross_entropy, accuracy


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
