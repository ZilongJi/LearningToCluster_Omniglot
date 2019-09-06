"""
python ./train/train_proto_hardtriplet.py --trainsize -1 --base_lr 0.005 --rho 0.6e-3

testing in 5-way 1-shot mode: 
    python ./train/train_proto_hardtriplet.py --only_test True --test_way 5 --test_shot 1


"""

from __future__ import print_function, absolute_import
import os
import os.path as osp
import sys
sys.path.append('./')
import time
import pdb
import argparse
import random
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import *
from src.evaluators import extract_features
from src.Model import SimpleCNN_PTN
from src.rerank import re_ranking
from src import dataset
from src.loss import global_loss
from src.TripletLoss import TripletLoss
from src.sampler import RandomIdentitySampler

from protonets.prototypical_batch_sampler import PrototypicalBatchSampler
from protonets.prototypical_loss import prototypical_loss as loss_fn

cudnn.deterministic = True # cudnn  
cudnn.benchmark = False 

def get_fewshot_testdata(args):
    dat_set = dataset.Omniglot(root=args.data_dir, 
        train=False,
        transform=transforms.Compose([
            transforms.Resize(32),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
    
    classes_per_it = args.test_way
    num_samples = args.test_query+args.test_shot
    sampler = PrototypicalBatchSampler(labels=dat_set.labels,
        classes_per_it=classes_per_it,
        num_samples=num_samples,
        iterations=args.test_episodes)
    dataloader = torch.utils.data.DataLoader(dat_set, batch_sampler=sampler) 
    return dataloader   

def train(args, model, device, optimizer, tri_loss, exp_dir):
    #change rho avlue accoding to training numbers
    if args.trainsize > 2000 and args.trainsize <= 6000:
        rho = 1.7e-3
    elif args.trainsize > 6000 and args.trainsize <= 8000:
        rho = 1.5e-3
    elif args.trainsize > 8000 and args.trainsize <=10000:
        rho = 1.3e-3
    elif args.trainsize > 10000 and args.trainsize <=12000:
        rho = 1.1e-3    
    elif args.trainsize > 12000 and args.trainsize <=14000:
        rho = 0.9e-3           
    elif args.trainsize > 14000 and args.trainsize <=16000:
        rho = 0.7e-3 
    else:
        rho = args.rho 
            
    #start episodic training
    total_NMI = np.zeros(args.iteration)
    total_AMI = np.zeros(args.iteration)
    total_SMI = np.zeros(args.iteration)
    total_ACCU = np.zeros(args.iteration+1)
    
    #Tesing before self-training
    print('Tesing before self-training')
    accu = test(args, model, device)
    total_ACCU[0] = accu    
    
    for iter_n in range(args.iteration): 
    
        #generate data loader
        extraction_loader = DataLoader(
            dataset.Omniglot(root=args.data_dir, 
                train=True, 
                size=args.trainsize, 
                transform=transforms.Compose([
                    transforms.Resize(32),
                    #transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        #extract all data features
        train_features, target_labels = extract_features(model=model, data_loader=extraction_loader, device=device)    
                           
        #rerank to get the jaccard distance
        rerank_dist = re_ranking(features=train_features, MemorySave=args.memory_save)
        
        #build the DBSCAN model
        tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
        tri_mat = np.sort(tri_mat,axis=None)
        top_num = np.round(rho*tri_mat.size).astype(int)
        eps = tri_mat[:top_num].mean()
        print('eps in cluster: {:.3f}'.format(eps))
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)        
              
        # select & cluster images as training set of this episode
        print('Clustering and labeling...')
        train_features = train_features.cpu().numpy()
        labels = cluster.fit_predict(rerank_dist)               
        
        #calculate NMI of chosed data points of current episode
        TL = target_labels
        list_true = [int(TL[i].cpu().numpy()) for i in range(len(TL))]
        list_pred = labels.tolist()

        NMI = nmi_withGT(list_pred, list_true) 
        AMI = ami_withGT(list_pred, list_true)

        SMI = sampling_NMI_withGT(list_pred, list_true)       
        total_NMI[iter_n] = NMI
        total_AMI[iter_n] = AMI
        total_SMI[iter_n]  =SMI  
        
        num_ids = len(set(labels)) - 1            
        #generate new dataset
        new_dataset = []
        
        unique_labels, label_count = np.unique(labels, return_counts=True)

        for i in range(len(extraction_loader.dataset.splittxt)):
            idd = np.where(unique_labels==labels[i])[0][0]
            
            if labels[i]==-1 or label_count[idd]<6:
                continue

            new_dataset.append((extraction_loader.dataset.splittxt[i], labels[i], 0))        

        LL = [new_dataset[i][1] for i in range(len(new_dataset))]
        print(np.unique(LL, return_counts=True))

        print('Iteration {} have {} training ids, {} training images, NMI is {}, AMI is {}, SMI is {}'.format(iter_n+1, num_ids, len(new_dataset), NMI, AMI, SMI)) 

        #triplet_proto dataloader
        BS = args.batch_size * args.ims_per_id        
        train_loader = DataLoader(
            dataset.Omniglot_clustering(root=args.data_dir,
                dat_set=new_dataset, 
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
            batch_size=BS, num_workers=4,
            sampler=RandomIdentitySampler(new_dataset, args.ims_per_id), 
            pin_memory=True, drop_last=True)             
        
        #training with prototipical learning methods
        for ep in range(args.epochs):
            # Adjust Learning Rate 
            adjust_lr_exp(
                optimizer,
                args.base_lr,
                ep +1,
                args.epochs,
                args.exp_decay_at_epoch)             
            
            model.train() 
              
            protoacc_meter = AverageMeter()      
            protoloss_meter = AverageMeter() 
            triloss_meter =  AverageMeter()
            totalloss_meter = AverageMeter()
            ep_st = time.time()             

            for data, target in tqdm(train_loader):    
                #pdb.set_trace()
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()                
                feat, _ = model(data)

                protoloss, acc = loss_fn(feat, target=target, n_support=args.train_shot)
                protoloss = protoloss.to(device)     
                
                triloss, _, _, _, _, _ = global_loss(tri_loss, feat, target, normalize_feature=args.normalize_feature, hard_mining=args.hard_mining)               
                
                total_loss = protoloss + triloss
                total_loss.backward() 
                optimizer.step()                  

                protoacc_meter.update(acc.item())
                protoloss_meter.update(protoloss.item())
                triloss_meter.update(triloss.item())
                totalloss_meter.update(total_loss.item())

            #Epoch log
            time_log = 'Ep {}, {:.2f}s'.format(ep, time.time() - ep_st, )  
            
            loss_log = (', acc {:.2%}, protoloss {:.4f}, triloss {:.4f}, total loss {:.4f}'.format(protoacc_meter.avg, protoloss_meter.avg, triloss_meter.avg, totalloss_meter.avg))             
            
            final_log = time_log + loss_log
            
            print(final_log)      
        
        #adjust learning rate back to initialized learning rate
        print('Learning rate adjuested back to base learning rate {:.10f}'.format(args.base_lr))
        for g in optimizer.param_groups:
            g['lr'] = args.base_lr  
            
        accu = test(args, model, device)
        
        total_ACCU[iter_n+1] = accu            
            
    print('total NMI value is, ', total_NMI)  
    print('total AMI value is, ', total_AMI)
    print('total SMI value is, ', total_SMI)
    print('total ACCU value is, ', total_ACCU)              
                    
         
def test(args, model, device):
    print('*' * 50)
    print('Few Shot Learning Testing')
    print('*' * 50)
    model.eval()
    test_dataloader = get_fewshot_testdata(args)
    
    avg_acc = list()
    for epoch in range(5):
        print('current testing epoch', epoch)
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output, _ = model(x)
            _, acc = loss_fn(model_output, target=y, n_support=args.test_shot)
            avg_acc.append(acc.item())
    
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))  

    model.train()
    
    return avg_acc 

def main(args):
    np.random.seed(args.seed) #numpy
    torch.manual_seed(args.seed)  #cpu
    torch.cuda.manual_seed(args.seed)  #gpu  
    random.seed(args.seed) #random and transforms    

    if args.exp_dir == '':
        exp_dir = 'exp_proto_hardtriplet'

    if args.log_to_file:
        stdout_file = osp.join(exp_dir, 'stdout_{}.txt'.format(time_str()))
        stderr_file = osp.join(exp_dir,'stderr_{}.txt'.format(time_str()))
        ReDirectSTD(stdout_file, 'stdout', False)
        ReDirectSTD(stderr_file, 'stderr', False)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda:1" if use_cuda else "cpu")

    model = SimpleCNN_PTN().to(device)
    
    if args.only_test:
        #load parameters of trained model
        model.load_state_dict(torch.load(osp.join(exp_dir, "selftraining_model_"+str(args.trainsize)+".pt")))
        test(args, model, device)             
        return
        
    optimizer = optim.Adam(model.parameters(),
                           lr=args.base_lr,
                           weight_decay=0.0005)        
    
    tri_loss = TripletLoss(args.margin) 
        
    train(args, model, device, optimizer, tri_loss, exp_dir) 
    
    if (args.save_model):
        torch.save(model.state_dict(), exp_dir+"/selftraining_model_"+str(args.trainsize)+".pt")       
        
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Omniglot Example')
    parser.add_argument('--data_dir', type=str, metavar='PATH', default='/home/jizilong/Desktop/Clustering_Theory_Project/Clustering_Theory/Experiment/Dataset/omniglot')       
    parser.add_argument('--batch-size', type=int, default=60, metavar='N',
                        help='input batch size for training (default: 64), e.g. training ways')
    parser.add_argument('--ims_per_id', type=int, default=6, 
                        help="ims_per_id must >1 when using triplet loss, value=train shot + train query")
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train in a episode (default: 10)')
    parser.add_argument('--exp_decay_at_epoch', type=int, default=25) 
    parser.add_argument('--base_lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--memory_save', type=str2bool, default=False,
                        help='save memory when computing jaccard distance') 
    parser.add_argument('--only_test', type=str2bool, default=False, help='running code only in the testing mode')                        
    parser.add_argument('--iteration', type=int, default=20) 
    parser.add_argument('--rho', type=float, default=0.5e-3, help="rho percentage, default: 1.6e-3")  
    parser.add_argument('--margin', type=float, default=0.5, help="margin of the triplet loss, default: 0.3")
    parser.add_argument('--normalize_feature', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--train_shot', type=int, default=1, help="number of support examples per class during training")     
    parser.add_argument('--test_way', type=int, default=5, help="number of classes per episode in test")
    parser.add_argument('--test_shot', type=int, default=1, help="number of support examples per class in test")   
    parser.add_argument('--test_query', type=int, default=15, help="number of query examples per class in test") 
    parser.add_argument('--test_episodes', type=int, default=100, help="number of test episodes per epoch (default: 1000)")   
    parser.add_argument('--trainsize', type=int, default=-1, help="number of training examples") 
    parser.add_argument('--hard_mining', type=str2bool, default=True)

    args = parser.parse_args()    
    
    main(args)



