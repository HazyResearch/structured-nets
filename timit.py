import h5py
import scipy.io as sio
import numpy as np
import pickle as pkl

def process(feat_loc,lab_loc,train,top_N_classes=None,N=None):
	lab = sio.loadmat(lab_loc)['lab']
	if train:
		with h5py.File(feat_loc) as f:
			feat = np.array(f['fea'])
	else:
		feat = sio.loadmat(feat_loc)['fea']
	if top_N_classes is None:
		assert N is not None
		counts = np.bincount(lab.flatten())
		print('counts: ', counts)
		idx_array = np.argsort(counts)
		print('idx array: ', idx_array)
		top_N_classes = idx_array[-N:][::-1]
		print('top N classes: ', top_N_classes)
		print('top N counts: ', counts[top_N_classes])
		print('top N total: ', np.sum(counts[top_N_classes]))
	idx = np.array([i for i in range(lab.size) if lab[i] in top_N_classes])
	print('idx: ', idx.shape)

	return feat[idx,:],lab[idx], top_N_classes

N = 25
train_feat_loc = '../timit/timit_train_feat.mat'
train_lab_loc = '../timit/timit_train_lab.mat'
test_feat_loc = '../timit/timit_heldout_feat.mat'
test_lab_loc = '../timit/timit_heldout_lab.mat'
train_feat_out_loc = '../timit/timit_train_feat_top' + str(N) + '.p'
train_lab_out_loc = '../timit/timit_train_lab_top' + str(N) + '.p'
test_feat_out_loc = '../timit/timit_test_feat_top' + str(N) + '.p'
test_lab_out_loc = '../timit/timit_test_lab_top' + str(N) + '.p'

train_feat,train_lab, top_N_classes = process(train_feat_loc,train_lab_loc,True,top_N_classes=None,N=N)
test_feat,test_lab, _ = process(test_feat_loc,test_lab_loc,False,top_N_classes=top_N_classes,N=N)
print('train_feat,train_lab: ', train_feat.shape, train_lab.shape)
print('test_feat,test_lab: ', test_feat.shape, test_lab.shape)
print('train_lab: ', np.unique(train_lab))
print('test_lab: ', np.unique(test_lab))

# Dump
pkl.dump(train_feat, open(train_feat_out_loc, 'wb'),protocol=2)
pkl.dump(train_lab, open(train_lab_out_loc, 'wb'),protocol=2)
pkl.dump(test_feat, open(test_feat_out_loc, 'wb'),protocol=2)
pkl.dump(test_lab, open(test_lab_out_loc, 'wb'),protocol=2)
