import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter

from . import toeplitz as toep
from . import krylov as kry

# TODO: rewrite with structure.layer
# TODO: subclass with each DR type
class LDR(nn.Module):

    def name(self):
        return str(self.in_channels) + str(self.out_channels) + self.displacement + str(self.r)

    # TODO: support non-square multiplications
    def __init__(self, displacement, in_channels, out_channels, rank, layer_size, bias=True):
        super(LDR, self).__init__()
        self.displacement = displacement
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = rank
        self.n = layer_size
        self.bias = None

        self.G = Parameter(torch.Tensor(self.in_channels, self.out_channels, self.r, self.n))
        self.H = Parameter(torch.Tensor(self.in_channels, self.out_channels, self.r, self.n))
        torch.nn.init.normal_(self.G, std=0.01) #TODO
        torch.nn.init.normal_(self.H, std=0.01)
        if bias:
            self.bias = Parameter(torch.zeros(self.out_channels, 1, self.n))
        if self.displacement == 'toeplitz_corner' or self.displacement == 'tc':
            self.corner = True
        elif self.displacement == 'toeplitz' or self.displacement == 't':
            self.corner = False
        elif self.displacement == 'subdiagonal' or self.displacement == 'sd':
            self.subd_A = Parameter(torch.ones((self.in_channels, self.out_channels, self.n-1)))
            self.subd_B = Parameter(torch.ones((self.in_channels, self.out_channels, self.n-1)))


    def forward(self, x):
        """
        x: (in_channels, batch, n)
        out: (out_channels, batch, n)
        """
        _, b, n = x.shape
        assert n == self.n

        # print("shapes ", self.G[0,0].shape, self.H[0,0].shape, x[0].shape)
        comps = Variable(torch.Tensor(self.in_channels, self.out_channels, b, self.n)).cuda()
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                if self.displacement in ['toeplitz_corner', 'toeplitz', 'tc', 't']:
                    g = self.G[i,j]
                    h = self.H[i,j]
                    fat = x[i]
                    comps[i,j] = toep.toeplitz_mult(self.G[i,j], self.H[i,j], x[i], self.corner)
                elif self.displacement == 'subdiagonal' or self.displacement == 'sd':
                    comps[i,j] = kry.subdiag_mult_conv(self.subd_A[i,j], self.subd_B[i,j], self.G[i,j], self.H[i,j], x[i])
        out = torch.sum(comps, dim=0)
        if self.bias is not None:
            out += self.bias
        return out

    def loss(self):
        lamb = 0.0001
        # lamb = 0
        return lamb*torch.sum(torch.abs(self.G)) + lamb*torch.sum(torch.abs(self.H))
