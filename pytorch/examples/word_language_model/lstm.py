"""
Some parts modified from https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
"""

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import sys
sys.path.insert(0, '../../../pytorch/')
import structure.layer as sl

class LSTMCell(nn.Module):
    def __init__(self, class_type, r, input_size, hidden_size, use_bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.class_type = class_type
        self.r = r

        # Replace W_ih with structured matrices
        self.W_ih = sl.StructuredLinear(class_type, layer_size=4*hidden_size, r=r, bias=False)

        self.W_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        W_hh_data = torch.eye(self.hidden_size).repeat(1,4)
        self.W_hh.data.set_(W_hh_data)
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        h_0, c_0 = hx
        h_0 = h_0.squeeze()
        c_0 = c_0.squeeze()
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.W_hh)
        z = torch.zeros(input_.size(0),3*self.hidden_size).cuda()
        input_padded = torch.cat((input_, z), dim=1)
        wi = self.W_ih(input_padded)

        f, i, o, g = torch.split(wh_b + wi,
                                 split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

class SingleLayerLSTM(nn.Module):
    def __init__(self, class_type, r, input_size, hidden_size,use_bias=True,dropout=0):
        super(SingleLayerLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.dropout = dropout

        # Initialize LSTMCell
        self.cell = LSTMCell(class_type=class_type, r=r, input_size=input_size,
                              hidden_size=hidden_size, use_bias=use_bias)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask + hx[0]*(1 - mask)
            c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx):
        max_time, batch_size, _ = input_.size()
        length = Variable(torch.LongTensor([max_time] * batch_size))
        if input_.is_cuda:
            device = input_.get_device()
            length = length.cuda(device)

        output, (h_n, c_n) = SingleLayerLSTM._forward_rnn(
                cell=self.cell, input_=input_, length=length, hx=hx)
        input_ = self.dropout_layer(output)
        return output, (h_n, c_n)