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

path = 'SampleData/'

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

# Imagenet standards
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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


"""
Models
"""

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


load_model = True
path_model = 'Models/ver1/model_100.pth'
path_dict = 'Models/ver1/info_loss_acc.pkl'
if load_model == True:
    model_1 = torch.load(path_model)
    model_1.eval()
    with open(path_dict, 'rb') as f:
        info_loss_acc_1 = pickle.load(f)
num_test_images = len(test_images_tensor)
y_pred_1 = test_application(model_1, test_loader)

categories = {0:'Cloth Mask',1:'Mask Worn Incorrectly',2:'N-95 Mask',3:'No Face Mask',4:'Surgical Mask'}


for no, i in enumerate(y_pred_1):
    print(no, categories[i])
