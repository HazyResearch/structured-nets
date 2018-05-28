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
#from optimize_nmt import optimize_nmt
#from optimize_iwslt import optimize_iwslt
from optimize_vae import optimize_vae

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: get rid of params here
def test_split(net, dataloader, loss_name_fn):
    # assert data_X.shape[0] == data_Y.shape[0]
    n = len(dataloader.dataset)
    total_loss = 0.0
    total_acc = 0.0
    for data in dataloader:
        batch_X, batch_Y = data
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        # print("lengths: ", len(dataloader), len(batch_X), len(batch_Y))

        output = net(batch_X)
        loss_batch, acc_batch = compute_loss_and_accuracy(output, batch_Y, loss_name_fn)
        total_loss += len(batch_X)*loss_batch.data.item()
        total_acc += len(batch_X)*acc_batch.data.item()
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
    # steps_in_epoch = int(np.ceil(dataset.train_X.shape[0] / params.batch_size))
    lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    losses = {'Train': [], 'Val': [], 'DR': [], 'ratio': [], 'Test':[]}
    accuracies = {'Train': [], 'Val': [], 'Test':[]}

    # val_X, val_Y = Variable(torch.FloatTensor(dataset.val_X).cuda()), Variable(torch.FloatTensor(dataset.val_Y).cuda())
    best_val_acc = 0.0
    best_val_save = None

    loss_name_fn = get_loss(params) # tuple of (loss name, loss fn)


    def log_stats(name, split, loss, acc, step):
        losses[split].append(loss)
        accuracies[split].append(acc)
        writer.add_scalar(split+'/Loss', loss, step)
        writer.add_scalar(split+'/Accuracy', acc, step)
        print(f"{name} loss, accuracy {params.class_type}: {loss:.6f}, {acc:.6f}")


    # compute initial stats
    t1 = time.time()
    init_loss, init_accuracy = test_split(net, dataset.val_loader, loss_name_fn)
    log_stats('Initial', 'Val', init_loss, init_accuracy, 0)

    epochs = 0
    for epoch in range(1):
    # for step in range(1, params.steps+1):
        for step, data in enumerate(dataset.train_loader, 0):
        # get the inputs
            batch_xs, batch_ys = data
            batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

            # batch_xs, batch_ys = dataset.batch(params.batch_size, step)
            # batch_xs, batch_ys = Variable(torch.FloatTensor(batch_xs).cuda()), Variable(torch.FloatTensor(batch_ys).cuda())
            optimizer.zero_grad()   # zero the gradient buffers

            # output = net.forward(batch_xs)
            output = net(batch_xs)
            train_loss, train_accuracy = compute_loss_and_accuracy(output, batch_ys, loss_name_fn)
            train_loss += net.loss()
            train_loss.backward()

            optimizer.step()

            # log training every 100
            # params.log_freq?
            if step % 100 == 0:
                epochs += 1
                # lr_scheduler.step()

                print(('Time: ', time.time() - t1))
                t1 = time.time()
                print(('Training step: ', step))

                log_stats('Train', 'Train', train_loss.data.item(), train_accuracy.data.item(), step)

        # validate and checkpoint by epoch
        # Test on validation set
        val_loss, val_accuracy = test_split(net, dataset.val_loader, loss_name_fn)
        log_stats('Validation', 'Val', val_loss, val_accuracy, step)

        # record best model
        if val_accuracy > best_val_acc:
            # save_path = os.path.join(params.checkpoint_path, str(step))
            save_path = os.path.join(params.checkpoint_path, 'best')
            with open(save_path, 'wb') as f:
                torch.save(net.state_dict(), f)
            print(("Best model saved in file: %s" % save_path))

            best_val_acc = val_accuracy
            best_val_save = save_path

    # save last checkpoint
    save_path = os.path.join(params.checkpoint_path, 'last')
    with open(save_path, 'wb') as f:
        torch.save(net.state_dict(), f)
    print(("Last model saved in file: %s" % save_path))


    # Test trained model
    if params.test:
        # Load test
        # dataset.load_test_data()
        # test_X, test_Y = Variable(torch.FloatTensor(dataset.test_X).cuda()), Variable(torch.FloatTensor(dataset.test_Y).cuda())

        # Load net from best validation
        if best_val_save is not None: net.load_state_dict(torch.load(best_val_save))
        print(f'Loaded best validation checkpoint from: {best_val_save}')

        test_loss, test_accuracy = test_split(net, dataset.test_loader, loss_name_fn)
        log_stats('Test', 'Test', test_loss, test_accuracy, 0)

        # train_X, train_Y = Variable(torch.FloatTensor(dataset.train_X).cuda()), Variable(torch.FloatTensor(dataset.train_Y).cuda())
        train_loss, train_accuracy = test_split(net, dataset.train_loader, loss_name_fn)

        # Log best validation accuracy and training acc for that model
        writer.add_scalar('MaxAcc/Val', best_val_acc)
        writer.add_scalar('MaxAcc/Train', train_accuracy)


    writer.export_scalars_to_json(os.path.join(params.log_path, "all_scalars.json"))
    writer.close()

    return losses, accuracies
