import numpy as np
import os, sys
sys.path.insert(0, '../../pytorch/')
import torch
from torch_utils import *
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
sys.path.insert(0, '../../pytorch/attention/')
from attention import *
sys.path.insert(0, '../../')
from dataset import *

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        batch.src, batch.trg, batch.src_mask, batch.trg_mask = batch.src.cuda(), batch.trg.cuda(), batch.src_mask.cuda(), batch.trg_mask.cuda()
        #print('batch.src:', batch.src)
        #print('batch.trg: ', batch.trg)
        #print('batch.src_mask: ', batch.src_mask)
        #print('batch.trg_mask: ', batch.trg_mask)
        #quit()
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def optimize_nmt(dataset, params):
	V = 11
	criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
	model = make_model(params, V, V, N=2)
	for name, param in model.named_parameters():
	    if param.requires_grad:
	        print('Parameter name, shape: ', name, param.data.shape)

	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
	        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	model.cuda()

	for epoch in range(10):
	    model.train()
	    run_epoch(data_gen(V, 30, 20), model, 
	              SimpleLossCompute(model.generator, criterion, model_opt))
	    model.eval()
	    print(run_epoch(data_gen(V, 30, 5), model, 
	                    SimpleLossCompute(model.generator, criterion, None)))
