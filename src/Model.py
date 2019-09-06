import torch.nn as nn
import torch.nn.functional as F
import sys
from .resnet import resnet18

import pdb

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=None):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        feat = F.relu(self.fc1(x))
        x = self.fc2(feat)
        return feat, F.log_softmax(x, dim=1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class SimpleCNN_PTN(nn.Module):
    def __init__(self, num_classes=None):
        super(SimpleCNN_PTN, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2))
                
        self.encoder = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            Flatten()
        )
        
        #self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        #logit = F.log_softmax(self.fc(feat), dim=1)
        #logit = self.fc(feat)
        return feat, 0

class SimpleCNN_BN(nn.Module):
    def __init__(self, num_classes=None):
        super(SimpleCNN_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4*4*50, 50)
        self.global_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        pdb.set_trace()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        feat = self.global_bn(x)
        x = self.fc2(feat)
        return feat, F.log_softmax(x, dim=1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def ec_block(in_channels, out_channels):
            '''
            returns a block conv-bn-relu-pool
            '''
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2))
                
        self.encoder = nn.Sequential(
            ec_block(1, 64),
            ec_block(64, 64),
            ec_block(64, 64),
            ec_block(64, 64),
            Flatten()
        )

    def forward(self, x):
        feat = self.encoder(x)
        return feat
                
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        def dc_block(in_channels, out_channels):
            '''
            returns a block 
            '''
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ReLU(),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 3, padding=1))
        
        self.decoder = nn.Sequential(
            dc_block(64, 64),
            dc_block(64, 64),
            dc_block(64, 64),
            dc_block(64, 3)
        )
    
    def forward(self, x):
        x = x.view(x.shape[0], 64, 1, 1)
        feat = self.decoder(x)
        return feat

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        def ec_block(in_channels, out_channels):
            '''
            returns a block conv-bn-relu-pool
            '''
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2))
        
        def dc_block(in_channels, out_channels):
            '''
            returns a block 
            '''
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ReLU(),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
                
        self.encoder = nn.Sequential(
            ec_block(1, 64),
            ec_block(64, 64),
            ec_block(64, 64),
            ec_block(64, 64),
            Flatten()) 

        self.decoder = nn.Sequential(
            dc_block(64, 64),
            dc_block(64, 64),
            dc_block(64, 64),
            dc_block(64, 1)
        ) 
        
        self.global_bn = nn.BatchNorm1d(256) 
        
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.global_bn(feat)
        x  =  feat.view(feat.shape[0], 64, 2, 2)    
        x_hat = self.decoder(x)
        
        return feat, x_hat         

class AutoEncoder2(nn.Module):
    def __init__(self):
        super(AutoEncoder2, self).__init__()
        
        def ec_block(in_channels, out_channels):
            '''
            returns a block conv-bn-relu-pool
            '''
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2))
        
        def dc_block(in_channels, out_channels):
            '''
            returns a block 
            '''
            return nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(in_channels),
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1,output_padding=1))
                
        self.encoder = nn.Sequential(
            ec_block(1, 64),
            ec_block(64, 64),
            ec_block(64, 64),
            ec_block(64, 64),
            Flatten()) 

        self.decoder = nn.Sequential(
            dc_block(64, 64),
            dc_block(64, 64),
            dc_block(64, 64),
            dc_block(64, 1)
        ) 
        
        self.global_bn = nn.BatchNorm1d(256) 
        
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.global_bn(feat)
        x  =  feat.view(feat.shape[0], 64, 2, 2)   
        x_hat = self.decoder(x)
        
        return feat, x_hat
        
class Resnet18(nn.Module):
    def __init__(self, num_classes=None):
        super(Resnet18, self).__init__() 
        self.base = resnet18(pretrained=False)
        planes = 512
        self.fc1 = nn.Linear(planes, 64)
        self.global_bn = nn.BatchNorm1d(64) 
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.base(x)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        feat = self.global_bn(x)
        x = self.fc2(feat)
        return feat, F.log_softmax(x, dim=1)
        
class Resnet18_V2(nn.Module):
    def __init__(self, num_classes=None):
        super(Resnet18_V2, self).__init__() 
        self.base = resnet18(pretrained=True)
        planes = 512
        self.fc1 = nn.Linear(planes, 50)
        self.global_bn = nn.BatchNorm1d(50) 
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = self.base(x)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        feat = self.global_bn(x)
        x = self.fc2(feat)
        return feat, F.log_softmax(x, dim=1)        
        
