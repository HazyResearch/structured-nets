import numpy as np
import os, sys
sys.path.insert(0, '../../pytorch/')
import torch
from torch_utils import *
from torch.autograd import Variable
import torch.optim as optim
from torchtext import data, datasets
import spacy
from tensorboardX import SummaryWriter
sys.path.insert(0, '../../pytorch/attention/')
from attention import *
sys.path.insert(0, '../../')
from dataset import *

def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def create_src_tgt():
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    return SRC, TFT

def optimize_iwslt(dataset, params):
	SRC, TGT = create_src_tgt()
	print(len(SRC.vocab), len(TGT.vocab))
	quit()
	pad_idx = TGT.vocab.stoi["<blank>"]
	model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
	model.cuda()

	writer = SummaryWriter(params.log_path)
	losses = {'train': [], 'val': [], 'DR': [], 'ratio': []}
	accuracies = {'train': [], 'val': []}	

	criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
	for name, param in model.named_parameters():
	    if param.requires_grad:
	        print(('Parameter name, shape: ', name, param.data.shape))

	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
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