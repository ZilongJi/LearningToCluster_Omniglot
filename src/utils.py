from torch.autograd import Variable
from sklearn.manifold import TSNE
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
from sklearn.cluster import DBSCAN 
import datetime
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

import pdb

def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)

def time_str(fmt=None):
  if fmt is None:
    fmt = '%Y-%m-%d_%H:%M:%S'
  return datetime.datetime.today().strftime(fmt)

# Great idea from https://github.com/amdegroot/ssd.pytorch
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def to_scalar(vt):
  """Transform a length-1 pytorch Variable or Tensor to scalar. 
  Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]), 
  then npx = tx.cpu().numpy() has shape (1,), not 1."""
  if isinstance(vt, Variable):
    return vt.data.cpu().numpy().flatten()[0]
  if torch.is_tensor(vt):
    return vt.cpu().numpy().flatten()[0]
  raise TypeError('Input should be a variable or tensor')

class AverageMeter(object):
  """Modified from Tong Xiao's open-reid. 
  Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = float(self.sum) / (self.count + 1e-20)
    
def tsne_visual(features, labels, episode, n_components=2):
    """Visualize high-dimensional data in the feature space with t-distributed Stochastic Neighbor Embedding.
    
    Plotting part is from https://github.com/kevinzakka/tsne-viz/blob/master/main.py
    
    Args:
        features: features of all data points. shape (n_samples, n_features)
        labels: labels corresponding to all data points. shape (n_samples)
        episode: current testing episode
        n_components: Dimension of the embedded space.
    Returns:
        feat_embedded: Embedding of the training data in low-dimensional space. shape (n_samples, n_components)
    """     
    feat_embedded = TSNE(n_components=n_components).fit_transform(features)

    classes = np.unique(labels)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, len(classes)))
    
    xx = feat_embedded[:, 0]
    yy = feat_embedded[:, 1]
    
    #plot the images
    for i, class_i in enumerate(classes.tolist()):
        ax.scatter(xx[labels==class_i], yy[labels==class_i], color=colors[i], label=str(class_i), s=10)
    
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig('./tsne_figure/tsne_'+str(episode)+'.png', dpi=500)

class ReDirectSTD(object):
  """Modified from Tong Xiao's `Logger` in open-reid.
  This class overwrites sys.stdout or sys.stderr, so that console logs can
  also be written to file.
  Args:
    fpath: file path
    console: one of ['stdout', 'stderr']
    immediately_visible: If `False`, the file is opened only once and closed
      after exiting. In this case, the message written to file may not be
      immediately visible (Because the file handle is occupied by the
      program?). If `True`, each writing operation of the console will
      open, write to, and close the file. If your program has tons of writing
      operations, the cost of opening and closing file may be obvious. (?)
  Usage example:
    `ReDirectSTD('stdout.txt', 'stdout', False)`
    `ReDirectSTD('stderr.txt', 'stderr', False)`
  NOTE: File will be deleted if already existing. Log dir and file is created
    lazily -- if no message is written, the dir and file will not be created.
  """

  def __init__(self, fpath=None, console='stdout', immediately_visible=False):
    import sys
    import os
    import os.path as osp

    assert console in ['stdout', 'stderr']
    self.console = sys.stdout if console == 'stdout' else sys.stderr
    self.file = fpath
    self.f = None
    self.immediately_visible = immediately_visible
    if fpath is not None:
      # Remove existing log file.
      if osp.exists(fpath):
        os.remove(fpath)

    # Overwrite
    if console == 'stdout':
      sys.stdout = self
    else:
      sys.stderr = self

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
      may_make_dir(os.path.dirname(osp.abspath(self.file)))
      if self.immediately_visible:
        with open(self.file, 'a') as f:
          f.write(msg)
      else:
        if self.f is None:
          self.f = open(self.file, 'w')
        self.f.write(msg)

  def flush(self):
    self.console.flush()
    if self.f is not None:
      self.f.flush()
      import os
      os.fsync(self.f.fileno())

  def close(self):
    self.console.close()
    if self.f is not None:
      self.f.close()
    
def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    """
    Decay exponentially in the later phase of training. All parameters in the 
    optimizer share the same learning rate.
    """   
    assert ep >= 1, "Current epoch number should be >= 1"  

    if ep <= start_decay_at_ep:
        print('=====> lr stays the same as base_lr {:.10f}'.format(base_lr))
        return
    
    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                       / (total_ep + 1 - start_decay_at_ep))))  
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))      
    
def param_search(features):
    #EPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #MS = [2,3,4,5,6,7,8,9,10,15, 20, 30]     
    EPS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    MS = [2,3,4,5,6,7,8,9,10,15, 20, 30] 
    
    for eps in EPS:
        for min_samples in MS:
            cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=8)
            labels = cluster.fit_predict(features)
            
            LLL = []
            
            for i in range(len(labels)):
                if labels[i] == -1:
                    continue
                LLL.append(labels[i])
            
            num_ids = len(set(labels)) - 1
            print('EPS {} MS {} have {} training ids {} training images'.format(eps, min_samples, num_ids, len(LLL))) 
   
def param_search_dist(dist):
    EPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    MS = [2,3,4,5,6,7,8,9,10,15, 20, 30]       
    
    for eps in EPS:
        for min_samples in MS:
            cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=8)
            labels = cluster.fit_predict(dist)
            
            LLL = []
            
            for i in range(len(labels)):
                if labels[i] == -1:
                    continue
                LLL.append(labels[i])
            
            num_ids = len(set(labels)) - 1
            print('EPS {} MS {} have {} training ids {} training images'.format(eps, min_samples, num_ids, len(LLL)))     
    
def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)    
    
def nmi_withGT(list_pred, list_true):
    """
    calculate NMI between chosed data points and ground truth labels 
        list_pred: a list
        list_true: a list
    """
        
    labels_true = []
    labels_pred = []
    for i, label_i in enumerate(list_pred):
        if label_i == -1:
            continue
        labels_true.append(list_true[i])
        labels_pred.append(list_pred[i])
        
    NMI = normalized_mutual_info_score(labels_true, labels_pred) 
    
    return NMI 
    
def ami_withGT(list_pred, list_true):
    """
    calculate AMI between chosed data points and ground truth labels 
        list_pred: a list
        list_true: a list
    """
        
    labels_true = []
    labels_pred = []
    for i, label_i in enumerate(list_pred):
        if label_i == -1:
            continue
        labels_true.append(list_true[i])
        labels_pred.append(list_pred[i])
        
    AMI = adjusted_mutual_info_score(labels_true, labels_pred) 
    
    return AMI  

def help_(list_pred, list_true, sample_num):
    all_label = np.unique(list_true)
    select_labels = np.random.choice(all_label, size=sample_num, replace=False)
    index = []
    for label_i in select_labels:
        idx = np.where(np.asarray(list_true) == label_i)[0].tolist()
        index += idx
    select_label_pred = [list_pred[i] for i in index]
    select_label_true = [list_true[i] for i in index]
    
    return select_label_pred, select_label_true    

def sampling_NMI_withGT(list_pred, list_true, sample_num=10, epoch=50):
    """
    calculate NMI between chosed data points and ground truth labels by a sampling process
    """
    labels_true = []
    labels_pred = []
    for i, label_i in enumerate(list_pred):
        if label_i == -1:
            continue
        labels_true.append(list_true[i])
        labels_pred.append(list_pred[i])    

    select_NMI = []
    for epoch_i in range(epoch):
        #random select "sample_num" samples
        select_label_pred, select_label_true = help_(labels_pred, labels_true, sample_num)
        select_NMI_i = normalized_mutual_info_score(select_label_pred, select_label_true)
        select_NMI.append(select_NMI_i)
    
    return np.mean(select_NMI)        

    
    
    
    
    
    
    
    
    
       
