# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import pdb
import numpy as np

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def find_hardest(dist, target):
    """
    Find the hardest positive and negative examples of each query.
    Args:
        dist: pytorch tensor: N*N
    Returns:
        ind of each query: N*num_classes
    """
    dist = dist.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    N = dist.shape[0]
    classes = np.unique(target)
    n_classes = len(classes)   
    n_examples = np.sum(target==classes[0])
    
    final_ind = []
    pdb.set_trace()
    for i in range(N):
        
        IND = [0 for n in range(N)]
        
        dist2_query = dist[i]
    
        current_class = target[i]
        #find the hardest positive in the class

        k = np.where(classes==current_class)[0][0]
        pos_block = dist2_query[k*n_examples:(k+1)*n_examples]
        positive_idx = np.argmax(pos_block)
        PID = k*n_examples + positive_idx
        IND[PID] = 1
        
        #find the hardest negative in other classes 
        neg_classes = classes[classes!=current_class]
        for nc in neg_classes:
            k2 = np.where(classes==nc)[0][0]
            neg_block = dist2_query[k2*n_examples:(k2+1)*n_examples]
            negative_idx = np.argmin(neg_block)
            NID = k2*n_examples + negative_idx
            IND[NID] = 1        
        IND = np.asarray(IND)
        
        final_ind.append(IND)
        
    final_ind = np.vstack(final_ind)
    
    final_ind = torch.from_numpy(final_ind).byte()

    return final_ind, n_classes, n_examples   

def prototypical_loss_hem(input, target):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    ''' 
    #sort by labels
    target_sort, idx = torch.sort(target)
    input = input[idx]
    
    dists = euclidean_dist(input, input)    
    
    N = dists.size(0)
    
    ind, n_classes, n_examples = find_hardest(dists, target_sort)
    
    proto_dist = dists[ind].contiguous().view(N, -1)
    
    log_p_y = F.log_softmax(-proto_dist, dim=1)
    
    #create target labels
    #pdb.set_trace()
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1)
    target_inds = target_inds.expand(n_classes, n_examples).long()
    target_inds = target_inds.contiguous().view(N,1).cuda()
    
    loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(1)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    
    return loss_val,  acc_val
    
    
