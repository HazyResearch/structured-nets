import copy
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def get_train_valid_datasets(dataset,
                             valid_size=0.1,
                             random_seed=None,
                             shuffle=True):
    """
    Utility function for loading and returning train and validation
    datasets.
    Parameters:
    ------
    - dataset: the dataset, need to have train_data and train_labels attributes.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - random_seed: fix seed for reproducibility.
    - shuffle: whether to shuffle the train/validation indices.
    Returns:
    -------
    - train_dataset: training set.
    - valid_dataset: validation set.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    num_train = len(dataset)
    indices = list(range(num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_dataset, valid_dataset = copy.copy(dataset), copy.copy(dataset)
    train_dataset.train_data = train_dataset.train_data[train_idx]
    train_dataset.train_labels = train_dataset.train_labels[train_idx]
    valid_dataset.train_data = valid_dataset.train_data[valid_idx]
    valid_dataset.train_labels = valid_dataset.train_labels[valid_idx]
    return train_dataset, valid_dataset


def copy_with_new_transform(dataset, transform):
    """A copy of @dataset with its transform set to @transform.
    """
    new_dataset = copy.copy(dataset)
    new_dataset.transform = transform
    return new_dataset


def augment_transforms(augmentations, base_transform, add_id_transform=True):
    """Construct a new transform that stack all the augmentations.
    Parameters:
        augmentations: list of transforms (e.g. image rotations)
        base_transform: transform to be applied after augmentation (e.g. ToTensor)
        add_id_transform: whether to include the original image (i.e. identity transform) in the new transform.
    Return:
        a new transform that takes in a data point and applies all the
        augmentations, then stack the result.
    """
    if add_id_transform:
        fn = lambda x: torch.stack([base_transform(x)] + [base_transform(aug(x))
                                                     for aug in augmentations])
    else:
        fn = lambda x: torch.stack([base_transform(aug(x)) for aug in augmentations])
    return transforms.Lambda(fn)


def train(data_loader, model, optimizer):
    model.train()
    train_loss, train_acc = [], []
    for data, target in data_loader:
        if use_cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        pred = model.predict(output)
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()
        acc = (pred == target.data).sum() / target.data.size(0)
        train_loss.append(loss.data)
        train_acc.append(acc)
    return train_loss, train_acc


def train_models_compute_agreement(data_loader, models, optimizers):
    train_agreement = []
    for model in models:
        model.train()
    for data, target in data_loader:
        if use_cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
        pred, loss = [], []
        for model, optimizer in zip(models[:8], optimizers[:8]):
            optimizer.zero_grad()
            output = model(data)
            pred.append(model.predict(output))
            loss_minibatch = model.loss(output, target)
            loss_minibatch.backward()
            optimizer.step()
            loss.append(loss_minibatch.data)
        loss = np.array(loss)
        pred = np.array([p.cpu().numpy() for p in pred])
        train_agreement.append((pred == pred[0]).mean(axis=1))
    return train_agreement


def train_all_epochs(train_loader,
                     valid_loader,
                     model,
                     optimizer,
                     n_epochs,
                     verbose=True):
    model.train()
    train_loss, train_acc, valid_acc = [], [], []
    for epoch in range(n_epochs):
        if verbose:
            print(f'Train Epoch: {epoch}')
        loss, acc = train(train_loader, model, optimizer)
        train_loss += loss
        train_acc += acc
        correct, total = accuracy(valid_loader, model)
        valid_acc.append(correct / total)
        if verbose:
            print(
                f'Validation set: Accuracy: {correct}/{total} ({correct/total*100:.4f}%)'
            )
    return train_loss, train_acc, valid_acc


def accuracy(data_loader, model):
    """Accuracy over all mini-batches.
    """
    training = model.training
    model.eval()
    correct, total = 0, 0
    for data, target in data_loader:
        if use_cuda:
            data, target = Variable(
                data.cuda(), volatile=True), Variable(target.cuda())
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = model.predict(output)
        correct += (pred == target.data).sum()
        total += target.size(0)
    model.train(training)
    return correct, total


def all_losses(data_loader, model):
    """All losses over all mini-batches.
    """
    training = model.training
    model.eval()
    losses = []
    for data, target in data_loader:
        if use_cuda:
            data, target = Variable(
                data.cuda(), volatile=True), Variable(target.cuda())
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        losses.append([l.data[0] for l in model.all_losses(data, target)])
    model.train(training)
    return np.array(losses)


def agreement_kl_accuracy(data_loader, models):
    training = [model.training for model in models]
    for model in models:
        model.eval()
    valid_agreement, valid_acc, valid_kl = [], [], []
    for data, target in data_loader:
        if use_cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)
        pred, out = [], []
        for model in models:
            output = model(data)
            out.append(output)
            pred.append(model.predict(output))
        pred = torch.stack(pred)
        out = torch.stack(out)
        log_prob = F.log_softmax(out, dim=-1)
        prob = F.softmax(out[0], dim=-1).detach()
        valid_kl.append([F.kl_div(lp, prob, size_average=False).data.cpu() / prob.size(0) for lp in log_prob])
        valid_acc.append((pred == target.data).float().mean(dim=1).cpu().numpy())
        valid_agreement.append((pred == pred[0]).float().mean(dim=1).cpu().numpy())
    for model, training_ in zip(models, training):
        model.train(training_)
    return valid_agreement, valid_kl, valid_acc
