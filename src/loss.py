from __future__ import print_function
import torch
import numpy as np
import pdb

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input      
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    #pdb.set_trace()
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]c
        return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N]; 
          indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
          indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples, 
        thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
           .copy_(torch.arange(0, N).long())
           .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

#def gen_a_p_n(features, labels):
#    """
#    Generate anchor, positive, negative from "features" and "labels".
#    Args:
#    features: 
#    labels: pytorch LongTensor
#    Returns:
#    anchor: N x D, where N is the class number in a batch
#    positive: N x D, where N is the class number in a batch
#    negative: N x D, where N is the class number in a batch
#    """   
#    LA = labels.cpu().numpy()
#    unique_LA= np.unique(LA)
#    
#    anchor, positive, negative = [], [], []
#    #pdb.set_trace()
#    for ll in unique_LA:
#        ll_idx = np.where(LA==ll)[0]
#        #sampling anchors and positive example
#        an_idx, po_idx = np.random.choice(ll_idx,2,replace=False)
#        
#        #accoding to current anchor, sampling negative example
#        no_ll_idx = np.where(LA!=ll)[0]
#        neg_idx = np.random.choice(no_ll_idx,1,replace=False)[0]
#        
#        anchor.append(features[an_idx].unsqueeze(0))
#        positive.append(features[po_idx].unsqueeze(0))
#        negative.append(features[neg_idx].unsqueeze(0))
#        
#    anchor = torch.cat(anchor, dim=0)
#    positive = torch.cat(positive, dim=0)
#    negative = torch.cat(negative, dim=0)

#    return anchor, positive, negative

def gen_a_p_n(features, labels):
    """
    Generate anchor, positive, negative from "features" and "labels".
    Args:
    features: 
    labels: pytorch LongTensor
    Returns:
    anchor: N x D, where N is the example number in a batch
    positive: N x D, where N is the example number in a batch
    negative: N x D, where N is the example number in a batch
    """   
    LA = labels.cpu().numpy()
    unique_LA= np.unique(LA)
    
    anchor, positive, negative = [], [], []
    
    N = features.shape[0]
    
    for an_idx in range(N):
        an_label = LA[an_idx]
        all_po_idx = np.where(LA==an_label)[0]
        
        #random sampling a positive label from the remaining set sharing the same label with anchor
        all_po_idx = all_po_idx[all_po_idx!=an_idx]
        po_idx = np.random.choice(all_po_idx,1,replace=False)[0]
        
        #accoding to current anchor, sampling negative example
        all_neg_idx = np.where(LA!=an_label)[0]
        neg_idx = np.random.choice(all_neg_idx,1,replace=False)[0]
        
        anchor.append(features[an_idx].unsqueeze(0))
        positive.append(features[po_idx].unsqueeze(0))
        negative.append(features[neg_idx].unsqueeze(0))        
        
    anchor = torch.cat(anchor, dim=0)
    positive = torch.cat(positive, dim=0)
    negative = torch.cat(negative, dim=0)

    return anchor, positive, negative

#def gen_an_ap_dist(features, labels):
#    """
#    Generate anchor, positive, negative from "features" and "labels".
#    Args:
#    features: 
#    labels: pytorch LongTensor
#    Returns:
#    anchor: N x D, where N is the example number in a batch
#    positive: N x D, where N is the example number in a batch
#    negative: N x D, where N is the example number in a batch
#    """   
#    LA = labels.cpu().numpy()
#    unique_LA= np.unique(LA)
#    
#    anchor, positive, negative = [], [], []
#     
#    N = features.shape[0]
#    
#    #pdb.set_trace()
#    for an_idx in range(N):
#        an_label = LA[an_idx]
#        all_po_idx = np.where(LA==an_label)[0]
#        
#        #random sampling a positive label from the remaining set sharing the same label with anchor
#        all_po_idx = all_po_idx[all_po_idx!=an_idx]
#        po_idx = np.random.choice(all_po_idx,1,replace=False)[0]
#        
#        #accoding to current anchor, sampling negative example
#        all_neg_idx = np.where(LA!=an_label)[0]
#        neg_idx = np.random.choice(all_neg_idx,1,replace=False)[0]
#        
#        anchor.append(features[an_idx].unsqueeze(0))
#        positive.append(features[po_idx].unsqueeze(0))
#        negative.append(features[neg_idx].unsqueeze(0))  
#        
#    anchor = torch.cat(anchor, dim=0)
#    positive = torch.cat(positive, dim=0)
#    negative = torch.cat(negative, dim=0)

#    dist_ap = euclidean_dist(anchor, positive)
#    dist_an = euclidean_dist(anchor, negative)

#    dist_ap = torch.diag(dist_ap)
#    dist_an = torch.diag(dist_an)
#    
#    return dist_ap, dist_an 

def global_loss(tri_loss, features, labels, normalize_feature=True, hard_mining=True):
    """
    Args:
        tri_loss: a `TripletLoss` object
        features: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the 
            Channel dimension
    Returns:
        loss: pytorch Variable, with shape [1]
        p_inds: pytorch LongTensor, with shape [N]; 
          indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
          indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        =============
        For Debugging
        =============
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        ===================
    """  
    if normalize_feature:
        features = normalize(features, axis=-1)

    if hard_mining:
        # shape [N, N]
        dist_mat = euclidean_dist(features, features)      

        dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist_mat, labels, return_inds=True) 
        loss = tri_loss(dist_ap, dist_an) 
        return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat
    else:
        anchor, positive, negative = gen_a_p_n(features, labels)
        loss = tri_loss(anchor, positive, negative)
#        dist_ap, dist_an = gen_an_ap_dist(features, labels)
#        loss = tri_loss(dist_ap, dist_an)
        return loss, 0, 0, 0, 0, 0


def hard_example_mining_support(dist_mat, labels, support_idxs, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]c
        return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N]; 
          indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
          indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples, 
        thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    
    dist_ap = torch.stack([dist_ap[i] for i in support_idxs])
    dist_an = torch.stack([dist_an[i] for i in support_idxs])
    
    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
           .copy_(torch.arange(0, N).long())
           .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an
    
def global_loss_support(tri_loss, features, labels, support_idxs, query_idxs, normalize_feature=True, hard_mining=True):
    """
    Args:
        tri_loss: a `TripletLoss` object
        features: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the 
            Channel dimension
    Returns:
        loss: pytorch Variable, with shape [1]
        p_inds: pytorch LongTensor, with shape [N]; 
          indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
          indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        =============
        For Debugging
        =============
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        ===================
    """  
    if normalize_feature:
        features = normalize(features, axis=-1)

    if hard_mining:
        # shape [N, N]
        dist_mat = euclidean_dist(features, features)      

        dist_ap, dist_an, p_inds, n_inds = hard_example_mining_support(dist_mat, labels,  support_idxs, return_inds=True) 
        loss = tri_loss(dist_ap, dist_an) 
        return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat
    else:
        anchor, positive, negative = gen_a_p_n(features, labels)
        loss = tri_loss(anchor, positive, negative)
#        dist_ap, dist_an = gen_an_ap_dist(features, labels)
#        loss = tri_loss(dist_ap, dist_an)
        return loss, 0, 0, 0, 0, 0    
      
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
     
