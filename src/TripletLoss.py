import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

#class TripletLoss(object):
#  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
#  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
#  Loss for Person Re-Identification'."""
#  def __init__(self, margin=None):
#    self.margin = margin
#    if margin is not None:
#      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#    else:
#      self.ranking_loss = nn.SoftMarginLoss()

#  def __call__(self, dist_ap, dist_an):
#    """
#    Args:
#      dist_ap: pytorch Variable, distance between anchor and positive sample, 
#        shape [N]
#      dist_an: pytorch Variable, distance between anchor and negative sample, 
#        shape [N]
#    Returns:
#      loss: pytorch Variable, with shape [1]
#    """
#    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
#    if self.margin is not None:
#      loss = self.ranking_loss(dist_an, dist_ap, y)
#    else:
#      loss = self.ranking_loss(dist_an - dist_ap, y)
#    return loss

class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None):
    self.margin = margin
    self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample, 
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample, 
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    mm = torch.from_numpy(np.asarray(self.margin))
    loss = self.ranking_loss(dist_an - dist_ap-mm, y)
    return loss

#class TripletLoss(object):
#    def __init__(self, margin=None):
#        self.margin = margin
#        self.ranking_loss = nn.TripletMarginLoss(margin=margin, p=2)
#    def __call__(self, anchor, positive, negative):
#        loss = self.ranking_loss(anchor, positive, negative)
#        return loss
