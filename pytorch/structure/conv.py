import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter

import structure.layer as sl



# TODO: rewrite with structure.layer
# TODO: subclass with each DR type
class BConv2d(nn.Module):

    def name(self):
        return 'bconv' + str(self.in_channels) + '-' + str(self.out_channels)

    def construct_block(self, complex):
        return sl.Butterfly(complex=complex)
        # components = [[butterfly.BlockPermProduct(size=self.in_size, complex=self.complex, share_logit=True),butterfly.Block2x2DiagProduct(size=self.in_size, complex=self.complex)] for _ in range(self.depth)]
        # components_ = [l for lis in components for l in lis]
        # return nn.Sequential(*components_)


    # TODO: support non-square multiplications
    def __init__(self, in_channels, out_channels, in_size, out_size, bias=True, complex=True):
        super(BConv2d, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.in_size            = in_size
        self.out_size     = out_size
        self.bias         = None
        self.depth        = 2
        self.complex      =complex

        # assert out_size == in_size // 4 # only support this for now
        # self.shrink = in_size // out_size
        # self.shrink = 1

        # channels = [construct_block(self.complex) for _ in range(self.in_channels) for _ in range(self.out_channels/self.shrink)]
        self.channels = nn.ModuleList([nn.ModuleList([sl.Butterfly(layer_size=self.in_size, complex=self.complex, fixed_perm=True) for _ in range(self.out_channels)]) for _ in range(self.in_channels)])
        if bias:
            self.bias = torch.zeros(self.out_channels, 1, self.in_size)


    def forward(self, x):
        """
        x: (batch, in_channels, in_size**.5, in_size**.5)
        out: (batch, out_channels, out_size)
        """
        b, ic, h, w = x.shape
        assert h*w == self.in_size

        # x = x.view(b, ic, -1)
        x = x.view(ic, b, -1)

        # print("shapes ", self.G[0,0].shape, self.H[0,0].shape, x[0].shape)
        comps = torch.empty((self.in_channels, self.out_channels, b, self.in_size), device=x.device)
        for i in range(self.in_channels):
            # for j in range(self.out_channels/self.shrink):
            for j in range(self.out_channels):
                comps[i,j] = self.channels[i][j](x[i])
                # for k in range(self.shrink):
                    # comps[i,self.shrink*j+k] = result[i,j,:,::self.shrink]

                #comps[i,j] =  #toep.toeplitz_mult(self.G[i,j], self.H[i,j], x[i], self.corner)
        out = torch.mean(comps, dim=0)
        if self.bias is not None:
            out += self.bias
        # return out.view(b, self.out_channels, h/2, w/2)
        return out.view(b, self.out_channels, h, w)

    def loss(self):
        return 0
        lamb = 0.0001
        # lamb = 0
        return lamb*torch.sum(torch.abs(self.G)) + lamb*torch.sum(torch.abs(self.H))
