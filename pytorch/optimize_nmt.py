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

def run_epoch(writer, data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    losses = []
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
            print(("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed)))
            start = time.time()
            tokens = 0
            losses.append(loss/batch.ntokens)
    return total_loss / total_tokens, losses

def optimize_nmt(dataset, params):
	V = 11
	writer = SummaryWriter(params.log_path)
	losses = {'train': [], 'val': [], 'DR': [], 'ratio': []}
	accuracies = {'train': [], 'val': []}	

	criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
	model = make_model(params, V, V, N=2)
	for name, param in model.named_parameters():
	    if param.requires_grad:
	        print(('Parameter name, shape: ', name, param.data.shape))

	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
	        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	model.cuda()

	for epoch in range(params.steps):
            model.train()
            train_total_loss_fraction, train_losses = run_epoch(writer, data_gen(V, 30, 20), model, 
	              SimpleLossCompute(model.generator, criterion, model_opt))
            losses['train'] += train_losses
            print('Train, epoch: ', train_total_loss_fraction, epoch)
            writer.add_scalar('Train/Loss', train_total_loss_fraction, epoch)
            model.eval()
            val_total_loss_fraction, val_losses = run_epoch(writer, data_gen(V, 30, 5), model, 
	                    SimpleLossCompute(model.generator, criterion, None))
            losses['val'] += val_losses
            print('Val, epoch: ', val_total_loss_fraction, epoch)
            writer.add_scalar('Val/Loss', val_total_loss_fraction, epoch)

            # Checkpoint
            save_path = os.path.join(params.checkpoint_path, str(epoch))
            with open(save_path, 'wb') as f: 
                torch.save(model.state_dict(), f)
            print(("Model saved in file: %s" % save_path))


	writer.export_scalars_to_json(os.path.join(params.log_path, "all_scalars.json"))
	writer.close()
	return losses, accuracies
