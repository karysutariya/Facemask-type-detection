"""
## Import Libraries

All all libraries must be installed which have been imported in the next code block.
"""

## Importing libraries
import os
import numpy as np
import pickle
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from utils import train_val

## If old model: uncomment the next line and comment the subsquent next line
#model_name ='old_model_cross_validation'
model_name = 'resnet18'


"""## Models"""

## Old Model

class net_ver1(nn.Module):
    def __init__(self):
        super(net_ver1, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(28 * 28 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 5)
        )

        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim = 1)
    

## New Model: Resnet18

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
    
    
class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=5):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = input.view(input.shape[0], -1)
        input = self.fc(input)

        return input
    
    
#resnet18 = ResNet18(3, ResBlock, outputs=5)
#print('Summary of New Model (Resnet18: ',summary(resnet18, (3, 224, 224)))

## Training Models

k=10
splits = KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}
batch_size_train = 64
batch_size_val = 64

"""## Loading Data"""

saved_tensors_path = 'saved tensors/'
train_images_tensor = torch.load(saved_tensors_path+'train_images_tensor.pt')
train_labels_tensor = torch.load(saved_tensors_path+'train_labels_tensor.pt')

class CreateDataset:
    def __init__(self, images, labels):
        
        self.images = images
        self.labels = labels
    
    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]
        
        return image, label
    
    def __len__(self):
        return len(self.labels)

train_dataset = CreateDataset(images=train_images_tensor, labels=train_labels_tensor)

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):

    print('Fold {}'.format(fold + 1))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size_train,
                                               sampler = train_sampler,
                                               drop_last=False)

    val_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size_val,
                                               sampler=val_sampler,
                                               drop_last=False)    

    
    ## If old model: uncomment the next line and comment the subsquent next line
    #model_crossval = old_model()
    model_crossval = ResNet18(3, ResBlock, outputs=5)
    
    learning_rate = 0.001
    num_epochs_train = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_crossval.parameters(), lr=learning_rate)
    
    path = '../Models/'+model_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    model_crossval_info = {'model':model_crossval, 'optimizer':optimizer, 'criterion':criterion, 'path':path}

    model_crossval, info_loss_acc_cross_val = train_val(model_crossval_info, train_loader, val_loader, num_epochs_train, save_model = False)

    foldperf['fold{}'.format(fold+1)] = info_loss_acc_cross_val  
    
    # Saving the model
    torch.save(model_crossval, path+'model_cross_val_'+str(fold+1)+'.pth')
    
    
    with open(path+'history_fold.pkl', 'wb') as f:
        pickle.dump(foldperf, f)    
