"""
Modified from pytorch/examples/vae. Original license shown below:

Copyright (c) 2017,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function
from tensorboardX import SummaryWriter
import argparse, os
import torch,torchvision
import torch.utils.data
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from structured_layer import StructuredLinear
from torch.nn.modules.padding import ConstantPad2d

# Takes batch_size x 1 x 28 x 28 -> batch_size x 1 x 32 x 32 (pad with zeros)
def pad(data):
    pad_fn = ConstantPad2d(2,0)
    padded = pad_fn(data)
    return padded

class VAE(nn.Module):
    # tie_layers_same: A->A, B->B of the two layers
    # tie_layers_opp: A->B, B->A of the two layers
    def __init__(self, params=None, cuda=True):
        super(VAE, self).__init__()
        self.params = params
        self.fc1 = StructuredLinear(params)#nn.Linear(params.layer_size, params.layer_size) #
        self.fc21 = nn.Linear(self.params.layer_size, 20)
        self.fc22 = nn.Linear(self.params.layer_size, 20)
        if self.params.num_structured_layers == 1:
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, self.params.layer_size)
        else:
            assert self.params.num_structured_layers == 2
            self.fc3 = nn.Linear(20, self.params.layer_size)
            self.fc4 = StructuredLinear(params)
            if self.params.class_type == 'subdiagonal':
                if self.params.tie_layers_A_A:
                    self.fc4.subd_A = self.fc1.subd_A
                    self.fc4.subd_B = self.fc1.subd_B
                elif params.tie_layers_A_B:
                    self.fc4.subd_A = self.fc1.subd_B
                    self.fc4.subd_B = self.fc1.subd_A

        self.device = torch.device("cuda" if cuda else "cpu")

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.params.layer_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, params):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, params.layer_size), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(model,epoch,dataset,optimizer,params):
    model.train()
    train_loss = 0
    n_batches = 200
    total_examples = n_batches*params.batch_size
    for batch_idx in range(n_batches):
        data, _ = dataset.batch(params.batch_size, batch_idx)
        data = torch.Tensor(data.reshape((data.shape[0],1,28,28)))
        data = pad(data)
        data = data.to(model.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar,params)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_train_loss = train_loss / total_examples
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, avg_train_loss))
    return avg_train_loss

def validate(model,epoch,dataset,params,test=False):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        data = dataset.test_X if test else dataset.val_X
        data = torch.Tensor(data.reshape((data.shape[0],1,28,28)))
        data = pad(data)
        data = data.to(model.device)
        recon_batch, mu, logvar = model(data)
        val_loss += loss_function(recon_batch, data, mu, logvar, params).item()

    val_loss /= data.shape[0]
    prefix = 'Test' if test else 'Validation'
    print('====> ' + prefix + ' loss: {:.4f}'.format(val_loss))
    return val_loss

def optimize_vae(dataset, params, seed=1):
    torch.manual_seed(seed)
    model = VAE(params)
    model = model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    writer = SummaryWriter(params.log_path)
    print((torch.cuda.get_device_name(0)))

    def log_stats(name, split, loss, step):
        losses[split].append(loss)
        writer.add_scalar(split+'/Loss', loss, step)
        print(f"{name} loss  {params.class_type}: {loss:.6f}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(('Parameter name, shape: ', name, param.data.shape))

    losses = {'Train': [], 'Val': [], 'DR': [], 'ratio': [], 'Test':[], 'Best_val': np.inf,
        'Best_val_epoch': 0, 'Best_val_save': None}
    accuracies = {}

    print('Running for %s epochs' % params.steps)
    for epoch in range(1, params.steps + 1):
        train_loss = train(model,epoch,dataset,optimizer, params)
        val_loss = validate(model,epoch,dataset,params)

        # Log
        log_stats('Train', 'Train', train_loss, epoch)
        log_stats('Validation', 'Val', val_loss, epoch)

        # Update best validation loss so far
        if val_loss < losses['Best_val']:
            losses['Best_val'] = val_loss
            losses['Best_val_epoch'] = epoch
            save_path = os.path.join(params.checkpoint_path, str(epoch))
            losses['Best_val_save'] = save_path
            with open(save_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print(("Model saved in file: %s" % save_path))

        with torch.no_grad():
            sample = torch.randn(64, 20).to(model.device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 32, 32),
                       params.result_path + params.class_type + 'sample_' + str(epoch) + '.png')

    # Test
    if params.test:
        dataset.load_test_data()
        # Load net from best validation
        if losses['Best_val_save'] is not None: model.load_state_dict(torch.load(losses['Best_val_save']))
        print('Loaded best validation checkpoint from:', {losses['Best_val_save']})

        test_loss = validate(model,epoch,dataset,params,test=True)
        log_stats('Test', 'Test', test_loss, 0)

    writer.export_scalars_to_json(os.path.join(params.log_path, "all_scalars.json"))
    writer.close()

    return losses, accuracies
