import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch.nn.functional as F


import matplotlib.pyplot as plt
import seaborn as sns

## If old model: uncomment the next line and comment the subsquent next line
#model_name ='old_model_cross_validation'
#model_name ='old_model_train_test_split'
model_name = 'resnet18'

path = '../Data/SampleData/'

data = []
 

for img in os.listdir(path):
    img_path = os.path.join(path, img)
    try:
        image = Image.open(img_path)
        data.append(image)
    except:
        pass  

"""
Images need to be in same shape to feed it to CNN so those are transformed. 
"""

saved_tensors_path = 'saved tensors/'

mean = torch.load(saved_tensors_path+'mean_train.pt')
std = torch.load(saved_tensors_path+'std_train.pt')

# Validation does not use augmentation
test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])


test_images = []
for i in range(len(data)):
    try:
        test_images.append(test_transforms(data[i]))
    except:
        pass


test_images_numpy = [t.numpy() for t in test_images]
test_images_tensor = torch.tensor(test_images_numpy)


"""
Importing data into Pytroch DataLoader
"""

class CreateDataset_application:
    def __init__(self, images):
        
        self.images = images
    
    def __getitem__(self, index):
        
        image = self.images[index]

        return image
    
    def __len__(self):
        return len(self.images)


test_dataset = CreateDataset_application(images=test_images_tensor)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                         shuffle=False,
                                         drop_last=False)


"""## Models"""

## Old Model

# Model 1

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
    
"""Test function"""

def test_application(model, test_loader):

    # Test:
    output_pred = []


    with torch.no_grad():
        for images in (test_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            output_pred.append(predicted[0].tolist())


    return output_pred


path_model = '../Models/'+model_name+'/'

if model_name == 'resnet18' or model_name == 'old_model_cross_validation':
    path_model = path_model + 'model_cross_val_10.pth'
elif model_name == 'old_model_train_test_split':
    path_model = path_model + 'ver1/model_100.pth'



model = torch.load(path_model)
model.eval()
num_test_images = len(test_images_tensor)
y_pred = test_application(model, test_loader)

categories = {0:'Cloth Mask',1:'Mask Worn Incorrectly',2:'N-95 Mask',3:'No Face Mask',4:'Surgical Mask'}


for no, i in enumerate(y_pred):
    print(no, categories[i])
