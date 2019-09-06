# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np
import pdb

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


def prototypical_loss(input, target, n_support):
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
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    y_hat = y_hat.squeeze()
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val

def find_support_query_idxs(input_cpu, target_cpu):

    N = input_cpu.size(0)
    
    #########
    #SUPPORT#
    #########
    
    #randomly pick a support point from input
    ind = np.random.randint(N)
    support_class = target_cpu[ind]  
    support_input = input_cpu[ind]
    
    classes = torch.unique(target_cpu)
    
    S_IND = [0 for c in classes]
    
    A1 = classes==support_class
    S_IND[A1.nonzero()[0]] = torch.tensor([ind])
    
    #find all the negative hardest as the negative prototype idxs 
    neg_classes = classes[classes!=support_class]  
    
    for c in neg_classes:
        #find all negative inputs of class c
        current_negidxs = target_cpu == c
        current_negs = input_cpu[current_negidxs]
        #calculate the min distance of chosed support and the negative inputs of class c
        dist = euclidean_dist(support_input.unsqueeze(0), current_negs)
        #find the min index of class c
        _, idx = torch.max(dist,1)
        neg_idx = current_negidxs.nonzero()[idx][0]
        
        A2 = classes==c
        S_IND[A2.nonzero()[0]] = neg_idx

    #######
    #QUERY#
    #######       
    N_IND = []
    for i,c in enumerate(classes):
        current_idx = target_cpu.eq(c).nonzero()
        query_idx = current_idx[current_idx!=S_IND[i]]
        N_IND.append(query_idx)
    
    N_IND = torch.stack(N_IND).view(-1)
    
    return S_IND, N_IND
   
def hard_prototypical_loss(input, target, n_support):
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
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    
    support_idxs, query_idxs = find_support_query_idxs(input_cpu, target_cpu)
    
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_samples = input.to('cpu')[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    y_hat = y_hat.squeeze()
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val

def prototypical_loss_support(input, target, n_support):
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
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    y_hat = y_hat.squeeze()
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val, support_idxs, query_idxs
