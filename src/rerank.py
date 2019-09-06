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

def re_ranking(features, k1=20, k2=6, MemorySave=False, Minibatch=2000):
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

    print('Starting re_ranking...')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index==i)[0]  
        k_reciprocal_index = forward_k_neigh_index[fi]   ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])  
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    #original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = [] 
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(all_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    return jaccard_dist

