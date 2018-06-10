import numpy as np

def normalize_data(data):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)

    return apply_normalization(data,mean,std), mean, std

def apply_normalization(data, mean, std):
    normalized = (data-mean)/std
    print('Apply normalization: mean, std: ', np.mean(normalized,axis=0), np.std(normalized,axis=0))
    return normalized

def standardize(data, max_val=255.0):
    scaled = data/max_val
    scaled = (scaled - 0.5)/0.5
    print('min, max: ', np.min(scaled), np.max(scaled))
    return scaled

