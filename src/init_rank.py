#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: Zilong
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import numpy as np
import pdb

def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist

def init_ranking(features, k1=20, k2=6, MemorySave=False, Minibatch=2000):
    '''Re_Ranking by k-reciprocal encoding
    Args:
        features: feature vectors of all training data points
        k1, k2: parameters, the original paper is (k1=20,k2=6)
        MemorySave: set to 'True' when using MemorySave mode
        Minibatch: avaliable when 'MemorySave' is 'True'
        
    '''
    all_num = features.shape[0]    
    features = features.cpu().numpy()
    #features = features.astype(np.float32)

    print('Computing original distance...')
    
    if MemorySave:
        original_dist = np.zeros(shape = [all_num, all_num], dtype=np.float32)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(features)[0]:
                part_dist = compute_dist(features[i:it], features)
                original_dist[i:it] = np.power(part_dist,2).astype(np.float32)
            else:
                part_dist = compute_dist(features[i:,], features)
                original_dist[i:,:] = np.power(part_dist,2).astype(np.float32) 
                break
            i = it
    else:
        original_dist = compute_dist(features,features).astype(np.float32)     
        original_dist = np.power(original_dist,2).astype(np.float32) 
        
    gallery_num = original_dist.shape[0] #gallery_num=all_num
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.  

    return initial_rank

