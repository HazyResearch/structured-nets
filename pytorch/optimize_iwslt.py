import numpy as np
import os, sys
sys.path.insert(0, '../../pytorch/')
import torch
from torch_utils import *
from torch.autograd import Variable
import torch.optim as optim
from torchtext import data, datasets
#import spacy
from tensorboardX import SummaryWriter
sys.path.insert(0, '../../pytorch/attention/')
from attention import *
sys.path.insert(0, '../../')
from dataset import *
import pickle as pkl

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

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

    return SRC, TGT, train, val, test

# Duplicated
def run_epoch(data_iter, model, loss_compute):
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

def optimize_iwslt(dataset, params):
	SRC, TGT, train, val, test = create_src_tgt()
	#print(len(SRC.vocab), len(TGT.vocab))
	#pkl.dump(SRC, open('/dfs/scratch1/thomasat/datasets/iwslt/src.p', 'wb'), protocol=2)	
	#pkl.dump(TGT, open('/dfs/scratch1/thomasat/datasets/iwslt/tgt.p', 'wb'), protocol=2)
	#pkl.dump(train, open('/dfs/scratch1/thomasat/datasets/iwslt/train.p', 'wb'), protocol=2)   
	#pkl.dump(val, open('/dfs/scratch1/thomasat/datasets/iwslt/val.p', 'wb'), protocol=2)   
	#pkl.dump(test, open('/dfs/scratch1/thomasat/datasets/iwslt/test.p', 'wb'), protocol=2)   
	#SRC = pkl.load(open('/dfs/scratch1/thomasat/datasets/iwslt/src.p', 'rb'))
	#TGT = pkl.load(open('/dfs/scratch1/thomasat/datasets/iwslt/tgt.p', 'rb'))
	pad_idx = TGT.vocab.stoi["<blank>"]
	model = make_model(params, len(SRC.vocab), len(TGT.vocab), N=6)
	model.cuda()

	print((torch.cuda.get_device_name(0)))

	for name, param in model.named_parameters():
	    if param.requires_grad:
	        print(('Parameter name, shape: ', name, param.data.shape))

	writer = SummaryWriter(params.log_path)
	losses = {'train': [], 'val': [], 'DR': [], 'ratio': []}
	accuracies = {'train': [], 'val': []}	
	criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
	criterion.cuda()
	BATCH_SIZE = 1024#1200#0
	train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
	valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)


	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	for epoch in range(params.steps):
		model.train()
		train_total_loss_fraction, train_losses = run_epoch((rebatch(pad_idx, b) for b in train_iter), 
			  model, 
			  SimpleLossCompute(model.generator, criterion, model_opt))
		losses['train'] += train_losses
		print('Train, epoch: ', train_total_loss_fraction, epoch)
		writer.add_scalar('Train/Loss', train_total_loss_fraction, epoch)
		model.eval()
		val_total_loss_fraction, val_losses = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
				  model, 
				  SimpleLossCompute(model.generator, criterion, None))
		losses['val'] += val_losses
		print('Val, epoch: ', val_total_loss_fraction, epoch)
		writer.add_scalar('Val/Loss', val_total_loss_fraction, epoch)
		print(loss)

	"""
	losses['train'] += train_losses
	print('Train, epoch: ', train_total_loss_fraction, epoch)
	writer.add_scalar('Train/Loss', train_total_loss_fraction, epoch)
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
	"""

	writer.export_scalars_to_json(os.path.join(params.log_path, "all_scalars.json"))
	writer.close()
	return losses, accuracies
