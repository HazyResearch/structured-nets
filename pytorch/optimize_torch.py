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

def optimize_torch(dataset, params):
    if params.model == 'Attention':
        if dataset.name == 'copy':
            return optimize_nmt(dataset, params)
        elif dataset.name == 'iwslt':
            return optimize_iwslt(dataset, params)

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

    losses = {'train': [], 'val': [], 'DR': [], 'ratio': []}
    accuracies = {'train': [], 'val': []}

    val_X, val_Y = Variable(torch.FloatTensor(dataset.val_X).cuda()), Variable(torch.FloatTensor(dataset.val_Y).cuda())


    loss_fn = get_loss(params)

    t1 = time.time()
    epochs = 0
    for step in range(params.steps):
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
            losses['train'].append(train_loss.data)
            accuracies['train'].append(train_accuracy.data)
            writer.add_scalar('Train/Loss', train_loss.data, step)
            writer.add_scalar('Train/Accuracy', train_accuracy.data, step)

            print(('Training step: ', step))

            # Test on validation set
            output = net.forward(val_X)
            val_loss, val_accuracy = compute_loss_and_accuracy(output, val_Y, params, loss_fn)

            writer.add_scalar('Val/Loss', val_loss.data, step)
            writer.add_scalar('Val/Accuracy', val_accuracy.data, step)

            losses['val'].append(val_loss.data)
            accuracies['val'].append(val_accuracy.data)

            print(('Train loss, accuracy for class %s: %f, %f' % (params.class_type, train_loss.data, train_accuracy.data)))
            print(('Validation loss, accuracy %s: %f, %f' % (params.class_type, val_loss.data, val_accuracy.data)))

        if step % params.checkpoint_freq == 0:
            save_path = os.path.join(params.checkpoint_path, str(step))
            with open(save_path, 'wb') as f:
                torch.save(net.state_dict(), f)
            print(("Model saved in file: %s" % save_path))

    # Test trained model
    if params.test:
        # Load test
        dataset.load_test_data()
        test_X, test_Y = Variable(torch.FloatTensor(dataset.test_X).cuda()), Variable(torch.FloatTensor(dataset.test_Y).cuda())

        output = net.forward(test_X)
        test_loss, test_accuracy = compute_loss_and_accuracy(output, test_Y, params, loss_fn)

        writer.add_scalar('Test/Loss', test_loss.data)
        writer.add_scalar('Test/Accuracy', test_accuracy.data)

        print(('Test loss, %s: %f' % (params.class_type, test_loss.data)))
        print(('Test accuracy, %s: %f ' % (params.class_type, test_accuracy.data)))

        losses['test'] = test_loss.data
        accuracies['test'] = test_accuracy.data


    writer.export_scalars_to_json(os.path.join(params.log_path, "all_scalars.json"))
    writer.close()

    return losses, accuracies
