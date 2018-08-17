import numpy as np
from learning import train
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_mask(net, prune_factor):
    weights = net.W.W.cpu().data.numpy()
    N = int(weights.size/prune_factor)
    # Get indices of N highest magnitude weights
    idx = np.abs(weights.flatten()).argsort()[-N:][::-1]

    Z = np.zeros(weights.size)
    Z[idx] = 1
    Z = Z.reshape(weights.shape)

    return Z

def prune(dataset, net, optimizer, lr_scheduler, epochs, log_freq, log_path, checkpoint_path, result_path, test, save, prune_lr_decay, prune_factor, prune_iters):
    # Initial training
    train.train(dataset, net, optimizer, lr_scheduler, epochs, log_freq, log_path, checkpoint_path, result_path, 0, save)

    for i in range(prune_iters):
        # Generate mask
        mask = generate_mask(net, prune_factor)

        # Set mask
        net.W.set_mask(mask, device)

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = prune_lr_decay*param_group['lr']

        # Retrain
        train.train(dataset, net, optimizer, lr_scheduler, epochs, log_freq, log_path, checkpoint_path, result_path, test, save, (i+1)*epochs)
