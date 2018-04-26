import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter

import toeplitz_gpu as toep
import krylov_multiply as subd

# TODO: subclass with each DR type
class LDR(nn.Module):
    # TODO: support non-square multiplications
    def __init__(self, displacement, in_channels, out_channels, rank, layer_size, bias=False):
        super(LDR, self).__init__()
        self.displacement = displacement
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = rank
        self.n = layer_size

        self.G = Parameter(torch.Tensor(self.in_channels, self.out_channels, self.r, self.n))
        self.H = Parameter(torch.Tensor(self.in_channels, self.out_channels, self.r, self.n))
        # self.b = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        torch.nn.init.normal(self.G, std=0.01) #TODO maybe set stddev to 0.1 or something?
        torch.nn.init.normal(self.H, std=0.01)
        # torch.nn.init.normal(self.b)
        if self.displacement == 'toep_corner':
            self.corner = True
        elif self.displacement == 'toep_nocorn':
            self.corner = False
        elif self.displacement == 'subdiagonal':
            self.subd_A = Parameter(torch.ones((self.in_channels, self.out_channels, params.layer_size)))
            self.subd_B = Parameter(torch.ones((self.in_channels, self.out_channels, params.layer_size)))


    def forward(self, x):
        """
        x: (in_channels, batch, n)
        out: (out_channels, batch, n)
        """
        # print("x has shape", x.shape)
        # print("x has type", type(x))
        _, b, n = x.shape
        assert n == self.n

        # print("shapes ", self.G[0,0].shape, self.H[0,0].shape, x[0].shape)
        # return Variable(toep.toeplitz_mult(self.G[0,0], self.H[0,0], x[0], False).data.view(1, b, n))
        # return toep.toeplitz_mult(self.G[0,0], self.H[0,0], x[0], False).view(1, b, n)

        comps = Variable(torch.Tensor(self.in_channels, self.out_channels, b, self.n)).cuda()
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                if self.displacement in ['toep_corner', 'toep_nocorn']:
                    comps[i,j] = toep.toeplitz_mult(self.G[i,j], self.H[i,j], x[i], self.corner)
                elif self.displacement == 'subdiagonal':
                    comps[i,j] = subd.subd_mult(self.subd_A[i,j], self.subd_B[i,j], self.G[i,j], self.H[i,j], x[i])
        # return Variable(torch.sum(comps, dim=0))
        return torch.sum(comps, dim=0)

    def loss(self):
        lamb = 0.00001
        # lamb = 0
        return lamb*torch.sum(torch.abs(self.G)) + lamb*torch.sum(torch.abs(self.H))
