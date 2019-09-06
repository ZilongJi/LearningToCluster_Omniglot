from __future__ import print_function
import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

import pdb

class MNIST(data.Dataset):
    """MNIST <http://yann.lecun.com/exdb/mnist/>_ Dataset.
    
    """

    training_file = 'training.pt'
    test_file = 'test.pt'
        
    def __init__(self, root, train=True, transform=None):
        self.root = root            
        self.transform = transform
        self.train = train  # training set or test set         
            
        if self.train:
            data_file = self.training_file           
        else:
            data_file = self.test_file   
        
        data, targets = torch.load(os.path.join(self.root, data_file))  
        N = len(data)   
        index = np.random.choice(np.arange(N), size=10000, replace=False)
        self.data, self.targets = data[index], targets[index]
                              
        #self.data, self.targets = torch.load(os.path.join(self.root, data_file))
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
    
        if self.transform is not None:
            img = self.transform(img)       

        return img, target
        
    def __len__(self):
        return len(self.data)  
       
class MNIST_Part(data.Dataset):
    
    training_file = 'training.pt'
    test_file = 'test.pt'
        
    def __init__(self, root, num, train=True, transform=None):
        self.root = root            
        self.transform = transform
        self.num = num # only works under training 
        self.train = train  # training set or test set         
            
        if self.train:
            data_file = self.training_file
            #random samlping #num# examples with uniform distribution over all entries      
            data, targets = torch.load(os.path.join(self.root, data_file))
            N = len(data)    
            #index = np.random.choice(np.arange(N), size=num, replace=False)
            index = np.load('index.npy')
            #pdb.set_trace()
            self.data, self.targets = data[index], targets[index]
        else:
            data_file = self.test_file   
            self.data, self.targets = torch.load(os.path.join(self.root, data_file))
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
    
        if self.transform is not None:
            img = self.transform(img)         

        return img, target
        
    def __len__(self):
        return len(self.data)   
        
class MNIST_clustering(data.Dataset):
    def __init__(self, dat_set, transform=None):
        self.dat_set = dat_set
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.dat_set[index][0], self.dat_set[index][1]
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.dat_set)        
        
class Omniglot(data.Dataset):
    def __init__(self, root, train=True, size=-1, transform=None):
        splits = []
        labels = []
        if train:
            with open(osp.join(root, 'new_splits', 'train.txt'), 'r') as f:
                for line in f:
                    splits.append(line.split(' ')[0])
                    labels.append(int(line.split(' ')[1][:-1]))  
        else:
            with open(osp.join(root, 'new_splits', 'test.txt'), 'r') as f:
                for line in f:
                    splits.append(line.split(' ')[0])
                    labels.append(int(line.split(' ')[1][:-1]))      
        self.root = root
        N = len(splits)
        #pdb.set_trace()
        if train:
#            pdb.set_trace()
#            index = np.random.choice(np.arange(N), size=5000, replace=False)
#            index = np.arange(10000)
#            self.splittxt = [splits[i] for i in index]
#            self.labels = [labels[i] for i in index]
            #index = np.random.choice(np.arange(N), size=5000, replace=False)
            if size==-1:
                #using all training examples
                self.splittxt = splits
                self.labels = labels   
            else:            
                index = np.arange(size)
                self.splittxt = [splits[i] for i in index]
                self.labels = [labels[i] for i in index]
          
        else:
            #self.splittxt = [splits[i] for i in index]
            #self.labels = [labels[i] for i in index]
            self.splittxt = splits
            self.labels = labels
        
        self.transform = transform
        
    def __getitem__(self, index):
        imgpath = osp.join(self.root, 'data', self.splittxt[index])
        img = Image.open(imgpath)
        label = self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label 
    
    def __len__(self):
        return len(self.splittxt)

class Omniglot_AE(data.Dataset):
    def __init__(self, root, train=True, size=-1, transform1=None, transform2=None):
        splits = []
        labels = []
        if train:
            with open(osp.join(root, 'new_splits', 'train.txt'), 'r') as f:
                for line in f:
                    splits.append(line.split(' ')[0])
                    labels.append(int(line.split(' ')[1][:-1]))  
        else:
            with open(osp.join(root, 'new_splits', 'test.txt'), 'r') as f:
                for line in f:
                    splits.append(line.split(' ')[0])
                    labels.append(int(line.split(' ')[1][:-1]))      
        self.root = root
        N = len(splits)
        #pdb.set_trace()
        if train:
#            pdb.set_trace()
#            index = np.random.choice(np.arange(N), size=5000, replace=False)
#            index = np.arange(10000)
#            self.splittxt = [splits[i] for i in index]
#            self.labels = [labels[i] for i in index]
            #index = np.random.choice(np.arange(N), size=5000, replace=False)
            if size==-1:
                #using all training examples
                self.splittxt = splits
                self.labels = labels   
            else:            
                index = np.arange(size)
                self.splittxt = [splits[i] for i in index]
                self.labels = [labels[i] for i in index]
          
        else:
            #self.splittxt = [splits[i] for i in index]
            #self.labels = [labels[i] for i in index]
            self.splittxt = splits
            self.labels = labels
        
        self.transform1 = transform1
        self.transform2 = transform2
        
    def __getitem__(self, index):
        imgpath = osp.join(self.root, 'data', self.splittxt[index])
        img = Image.open(imgpath)
        label = self.labels[index]
        
        if self.transform1 is not None:
            img1 = self.transform1(img)

        if self.transform2 is not None:
            img2 = self.transform2(img)
        
        return img1, img2, label 
    
    def __len__(self):
        return len(self.splittxt)

class Omniglot_protonet(data.Dataset):
    def __init__(self, root, dat_set, transform=None):
        self.root = root
        self.dat_set = dat_set
        self.transform = transform
        self.y = tuple([dat_set[i][1] for i in range(len(dat_set))])
        
    def __getitem__(self, index):
        imgpath, label = osp.join(self.root, 'data', self.dat_set[index][0]), self.dat_set[index][1]
        img = Image.open(imgpath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.dat_set)
       
class Omniglot_clustering(data.Dataset):
    def __init__(self, root, dat_set, transform=None):
        self.root = root
        self.dat_set = dat_set
        self.transform = transform
    
    def __getitem__(self, index):
        imgpath, label = osp.join(self.root, 'data', self.dat_set[index][0]), self.dat_set[index][1]
        img = Image.open(imgpath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.dat_set)
        
class Omniglot_TSNE(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        dat_set = os.path.listdir(self.root)
        self.dat_set = dat_set
        
    def __getitem__(self, index):
        imgpath = osp.join(self.root, self.dat_set[index])
        img = Image.open(imgpath)
    
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def __len__(self):
        return len(self.dat_set)


























