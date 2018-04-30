import numpy as np
import os, sys
sys.path.insert(0, '../../pytorch/')
from nets import construct_net
import time
import torch
from torch_utils import *
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from optimize_nmt import optimize_nmt
from optimize_iwslt import optimize_iwslt
from optimize_vae import optimize_vae


def test_split(net, data_X, data_Y, params, loss_fn, batch_size=None):
    assert data_X.shape[0] == data_Y.shape[0]
    n = data_X.shape[0]
    if batch_size is None:
        batch_size = n

    total_loss = 0.0
    total_acc = 0.0
    for b in range(0, n, batch_size):
        b_ = min(b+batch_size, n)
        batch_X = data_X[b:b_, ...]
        batch_Y = data_Y[b:b_, ...]

        output = net.forward(batch_X)
        loss_batch, acc_batch = compute_loss_and_accuracy(output, batch_Y, params, loss_fn)
        total_loss += (b_-b)*loss_batch.data
        total_acc += (b_-b)*acc_batch.data
    return total_loss/n, total_acc/n


def optimize_torch(dataset, params, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    if params.model == 'Attention':
        if dataset.name == 'copy':
            return optimize_nmt(dataset, params)
        elif dataset.name == 'iwslt':
            return optimize_iwslt(dataset, params)
    elif params.model == 'VAE':
            return optimize_vae(dataset, params)

    writer = SummaryWriter(params.log_path)
    net = construct_net(params)
    net.cuda()

    print((torch.cuda.get_device_name(0)))

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(('Parameter name, shape: ', name, param.data.shape))

    optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=params.mom)
    # optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=params.mom, weight_decay=1e-5)
    # optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    steps_in_epoch = int(np.ceil(dataset.train_X.shape[0] / params.batch_size))
    lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    losses = {'Train': [], 'Val': [], 'DR': [], 'ratio': [], 'Test':[]}
    accuracies = {'Train': [], 'Val': [], 'Test':[]}

    val_X, val_Y = Variable(torch.FloatTensor(dataset.val_X).cuda()), Variable(torch.FloatTensor(dataset.val_Y).cuda())
    best_val_acc = 0.0
    best_val_save = None

    loss_fn = get_loss(params)


    def log_stats(name, split, loss, acc, step):
        losses[split].append(loss)
        accuracies[split].append(acc)
        writer.add_scalar(split+'/Loss', loss, step)
        writer.add_scalar(split+'/Accuracy', acc, step)
        print(f"{name} loss, accuracy {params.class_type}: {loss:.6f}, {acc:.6f}")


    # compute initial stats
    t1 = time.time()
    init_loss, init_accuracy = test_split(net, val_X, val_Y, params, loss_fn, batch_size=params.batch_size)
    log_stats('Initial', 'Val', init_loss.item(), init_accuracy.item(), 0)

    epochs = 0
    for step in range(1, params.steps+1):
        batch_xs, batch_ys = dataset.batch(params.batch_size, step)
        batch_xs, batch_ys = Variable(torch.FloatTensor(batch_xs).cuda()), Variable(torch.FloatTensor(batch_ys).cuda())
        optimizer.zero_grad()   # zero the gradient buffers

        output = net.forward(batch_xs)
        train_loss, train_accuracy = compute_loss_and_accuracy(output, batch_ys, params, loss_fn)
        train_loss += net.loss()
        train_loss.backward()

        optimizer.step()

        if step % steps_in_epoch == 0:
            epochs += 1
            # lr_scheduler.step()

        # if step % params.test_freq == 0:
            print(('Time: ', time.time() - t1))
            t1 = time.time()
            print(('Training step: ', step))

            log_stats('Train', 'Train', train_loss.data.item(), train_accuracy.data.item(), step)

            # Test on validation set
            val_loss, val_accuracy = test_split(net, val_X, val_Y, params, loss_fn, batch_size=params.batch_size)
            log_stats('Validation', 'Val', val_loss.item(), val_accuracy.item(), step)

        # if step % params.checkpoint_freq == 0:
        # checkpoint by epoch
        if step % steps_in_epoch == 0:
            save_path = os.path.join(params.checkpoint_path, str(step))
            with open(save_path, 'wb') as f:
                torch.save(net.state_dict(), f)
            print(("Model saved in file: %s" % save_path))

            # record best model
            if val_accuracy.item() > best_val_acc:
                best_val_acc = val_accuracy.item()
                best_val_save = save_path

    # Test trained model
    if params.test:
        # Load test
        dataset.load_test_data()
        test_X, test_Y = Variable(torch.FloatTensor(dataset.test_X).cuda()), Variable(torch.FloatTensor(dataset.test_Y).cuda())

        # Load net from best validation
        if best_val_save is not None: net.load_state_dict(torch.load(best_val_save))
        print(f'Loaded best validation checkpoint from: {best_val_save}')

        test_loss, test_accuracy = test_split(net, test_X, test_Y, params, loss_fn, batch_size=params.batch_size)
        log_stats('Test', 'Test', test_loss.item(), test_accuracy.item(), 0)

        train_X, train_Y = Variable(torch.FloatTensor(dataset.train_X).cuda()), Variable(torch.FloatTensor(dataset.train_Y).cuda())
        train_loss, train_accuracy = test_split(net, train_X, train_Y, params, loss_fn, batch_size=params.batch_size)

        # Log best validation accuracy and training acc for that model
        writer.add_scalar('MaxAcc/Val', best_val_acc)
        writer.add_scalar('MaxAcc/Train', train_accuracy)


    writer.export_scalars_to_json(os.path.join(params.log_path, "all_scalars.json"))
    writer.close()

    return losses, accuracies
