import torch
from tqdm import tqdm
import pdb

def extract_features(model, data_loader, device):
    '''Extract features of all data points
    '''
    model.eval()
    
    feat_list = []
    label_list = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)): 
            data, target = data.to(device), target.to(device)
            feat, _ = model(data)
            feat_list.append(feat)
            label_list.append(target)
        features = torch.cat([feat_list[i] for i in range(len(feat_list))])
        labels = torch.cat([label_list[i] for i in range(len(label_list))])
    
    return features, labels
