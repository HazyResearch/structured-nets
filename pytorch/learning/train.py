import numpy as np
import os, time, logging
import pickle as pkl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_split(net, dataloader, loss_fn):
    n = len(dataloader.dataset)
    total_loss = 0.0
    total_acc = 0.0
    for data in dataloader:
        batch_X, batch_Y = data
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        output = net(batch_X)
        loss_batch, acc_batch = loss_fn(output, batch_Y)
        total_loss += len(batch_X)*loss_batch.data.item()
        total_acc += len(batch_X)*acc_batch.data.item()
    return total_loss/n, total_acc/n


# Epoch_offset: to ensure stats are not overwritten when called during pruning
def train(dataset, net, optimizer, lr_scheduler, epochs, log_freq, log_path, checkpoint_path, result_path,
    test, save_model, epoch_offset=0):
    logging.debug('Tensorboard log path: ' + log_path)
    logging.debug('Tensorboard checkpoint path: ' + checkpoint_path)
    logging.debug('Results directory: ' + result_path)

    os.makedirs(checkpoint_path, exist_ok=True)

    writer = SummaryWriter(log_path)
    net.to(device)

    logging.debug((torch.cuda.get_device_name(0)))

    for name, param in net.named_parameters():
        if param.requires_grad:
            logging.debug(('Parameter name, shape: ', name, param.data.shape))

    losses = {'Train': [], 'Val': [], 'DR': [], 'ratio': [], 'Test':[]}
    accuracies = {'Train': [], 'Val': [], 'Test':[]}

    best_val_acc = 0.0
    best_val_save = None

    # If not saving models, then keep updating test accuracy of best validation model
    test_acc_of_best_val = 0.0
    test_loss_of_best_val = 0.0

    def log_stats(name, split, loss, acc, step):
        losses[split].append(loss)
        accuracies[split].append(acc)
        writer.add_scalar(split+'/Loss', loss, step)
        writer.add_scalar(split+'/Accuracy', acc, step)
        logging.debug(f"{name} loss, accuracy: {loss:.6f}, {acc:.6f}")


    # Compute initial stats
    t1 = time.time()
    init_loss, init_accuracy = test_split(net, dataset.val_loader, dataset.loss)
    log_stats('Initial', 'Val', init_loss, init_accuracy, epoch_offset)

    for epoch in range(epochs):
        logging.debug('Starting epoch ' + str(epoch+epoch_offset))
        for step, data in enumerate(dataset.train_loader, 0):
            # Get the inputs
            batch_xs, batch_ys = data
            batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

            optimizer.zero_grad()   # Zero the gradient buffers

            output = net(batch_xs)
            train_loss, train_accuracy = dataset.loss(output, batch_ys)
            train_loss += net.loss()
            train_loss.backward()

            optimizer.step()

            # Log training every log_freq steps
            total_step = (epoch + epoch_offset)*len(dataset.train_loader) + step+1
            if total_step % log_freq == 0:
                logging.debug(('Time: ', time.time() - t1))
                t1 = time.time()
                logging.debug(('Training step: ', total_step))

                log_stats('Train', 'Train', train_loss.data.item(), train_accuracy.data.item(), total_step)

        # Validate and checkpoint by epoch
        # Test on validation set
        val_loss, val_accuracy = test_split(net, dataset.val_loader, dataset.loss)
        log_stats('Validation', 'Val', val_loss, val_accuracy, epoch+epoch_offset+1)

        # Update LR
        lr_scheduler.step()

        for param_group in optimizer.param_groups:
            logging.debug('Current LR: ' + str(param_group['lr']))

        # Record best model
        if val_accuracy > best_val_acc:
            if save_model:
                save_path = os.path.join(checkpoint_path, 'best')
                with open(save_path, 'wb') as f:
                    torch.save(net.state_dict(), f)
                logging.debug(("Best model saved in file: %s" % save_path))
                best_val_save = save_path

            else:
                test_loss, test_accuracy = test_split(net, dataset.test_loader, dataset.loss)
                test_loss_of_best_val = test_loss
                test_acc_of_best_val = test_accuracy


            best_val_acc = val_accuracy

    # Save last checkpoint
    if save_model:
        save_path = os.path.join(checkpoint_path, 'last')
        with open(save_path, 'wb') as f:
            torch.save(net.state_dict(), f)
        logging.debug(("Last model saved in file: %s" % save_path))

    # Test trained model
    if test:
        if save_model:
            # Load net from best validation
            if best_val_save is not None: net.load_state_dict(torch.load(best_val_save))
            logging.debug(f'Loaded best validation checkpoint from: {best_val_save}')

            test_loss, test_accuracy = test_split(net, dataset.test_loader, dataset.loss)
            log_stats('Test', 'Test', test_loss, test_accuracy, 0)

        else:
            log_stats('Test', 'Test', test_loss_of_best_val, test_acc_of_best_val, 0)

        train_loss, train_accuracy = test_split(net, dataset.train_loader, dataset.loss)

        # Log best validation accuracy and training acc for that model
        writer.add_scalar('MaxAcc/Val', best_val_acc)
        writer.add_scalar('MaxAcc/Train', train_accuracy)

    writer.export_scalars_to_json(os.path.join(log_path, "all_scalars.json"))
    writer.close()


    pkl.dump(losses, open(result_path + '_losses.p', 'wb'), protocol=2)
    pkl.dump(accuracies, open(result_path + '_accuracies.p', 'wb'), protocol=2)
    logging.debug('Saved losses and accuracies to: ' + result_path)

    return losses, accuracies
