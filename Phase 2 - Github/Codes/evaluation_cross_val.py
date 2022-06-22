import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch.nn.functional as F


import matplotlib.pyplot as plt
import seaborn as sns

from utils import test


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

    
## Load Test Loader
saved_tensors_path = 'saved tensors/'
test_loader = torch.load(saved_tensors_path+'test_loader.pth')
num_test_images = 0
for _, data in test_loader:
    num_test_images+=data.size()[0]

"""## Test Models
Download model from this link -  
Then, run the below code. 
"""


path_model = '../Models/'+model_name+'/'
path_dict = '../Models/'+model_name+'/history_fold.pkl'
fig_path = '../Figures/'+model_name+'/'

  
if not os.path.exists(fig_path):
  os.makedirs(fig_path)
  
with open(path_dict, 'rb') as f:
    history_fold = pickle.load(f)
    
g = open(path_model+'classification_report_'+model_name+'_folds.txt', "a")

categories = ['Cloth','Incorrectly', 'N95', 'NoMask', 'Surgical']
    
num_kfold = 10
class_report_macro_avg = {'precision':0,'recall':0,'f1-score':0,'support':0}
class_report_weighted_avg = {'precision':0,'recall':0,'f1-score':0,'support':0}
class_report_accuracy_avg = 0

for i in range(num_kfold):
    
    path = path_model+'model_cross_val_'+str(i+1)+'.pth'
    model_cross_val = torch.load(path)
    model_cross_val.eval()
    
    model_cross_val, y_pred_cross_val, y_true_cross_val, test_acc = test(model_cross_val, test_loader, num_test_images)
    
    g.write('Fold'+str(i+1)+': \n')
    g.write('Testing Accuracy of the model on the test images for fold'+str(i+1)+str(test_acc)+' % \n')
    g.write('Classification Report for the Whole dataset for fold'+str(i+1)+': \n')
    class_report = classification_report(y_true_cross_val, y_pred_cross_val, output_dict=True)
    g.write(classification_report(y_true_cross_val, y_pred_cross_val)+'\n')
    
    for key in class_report['macro avg']:
        class_report_macro_avg[key] += class_report['macro avg'][key]
        class_report_weighted_avg[key] += class_report['weighted avg'][key]
    class_report_accuracy_avg += class_report['accuracy']
    
    conf_mat = confusion_matrix(y_true_cross_val, y_pred_cross_val)
    df_cm = pd.DataFrame(conf_mat, categories, categories)
    plt.figure()
    sns_plot = sns.heatmap(df_cm, annot=True)
    sns_plot.figure.savefig(fig_path+'conf_mat_fold_'+str(i+1)+'.png')

for key in class_report_macro_avg:
    class_report_macro_avg[key]/=num_kfold    
    class_report_weighted_avg[key]/=num_kfold
class_report_accuracy_avg/=num_kfold
    
g.write('Aggregated Classification Report for the Whole dataset: \n')
g.write('Aggregated: class_report_macro_avg: \n'+str(class_report_macro_avg)+'\n')
g.write('Aggregated: class_report_weighted_avg: \n'+str(class_report_weighted_avg)+'\n')
g.write('Aggregated: Accuracy: '+str(class_report_accuracy_avg)+'\n')  
  
  
g.close()
    

h = open(path_model+'classification_report_'+model_name+'_subclass.txt', "a")

data_test = pd.read_csv('annotation.csv')
data_test.index+=1
data_test['True Label'] = y_true_cross_val
data_test['Predicted Label'] = y_pred_cross_val

## Whole Dataset

h.write('Testing Accuracy of the model on the test images: '+str(test_acc)+' % \n')
h.write('Classification Report for the Whole dataset: \n')
h.write(classification_report(y_true_cross_val, y_pred_cross_val))
conf_mat = confusion_matrix(y_true_cross_val, y_pred_cross_val)
# print('confusiton matrix: \n', conf_mat)
plt.figure()
df_cm = pd.DataFrame(conf_mat, categories, categories)
sns_plot = sns.heatmap(df_cm, annot=True)
sns_plot.figure.savefig(fig_path+'conf_mat_whole.png')


## Gender Analysis

## female
data_test_female = data_test.loc[data_test['Gender'] == 'female']
y_pred_female = data_test_female['Predicted Label']
y_true_female = data_test_female['True Label']

h.write('Classification Report for females in dataset: \n'+ classification_report(y_true_female, y_pred_female))
conf_mat_female = confusion_matrix(y_true_female, y_pred_female)
#print('confusiton matrix: \n', conf_mat_female)
plt.figure()
df_cm = pd.DataFrame(conf_mat_female, categories, categories)
sns_plot = sns.heatmap(df_cm, annot=True)
sns_plot.figure.savefig(fig_path+'conf_mat_female.png')


## male
data_test_male = data_test.loc[data_test['Gender'] == 'male']
y_pred_male = data_test_male['Predicted Label']
y_true_male = data_test_male['True Label']

h.write('Classification Report for male in dataset: \n'+classification_report(y_true_male, y_pred_male))
conf_mat_male = confusion_matrix(y_true_male, y_pred_male)
#print('confusiton matrix: \n', conf_mat_male)
plt.figure()
df_cm = pd.DataFrame(conf_mat_male, categories, categories)
sns_plot = sns.heatmap(df_cm, annot=True)
sns_plot.figure.savefig(fig_path+'conf_mat_male.png')

## Age Analysis 

## young
data_test_young = data_test.loc[data_test['Age'] == 'young']
y_pred_young = data_test_young['Predicted Label']
y_true_young = data_test_young['True Label']

h.write('Classification Report for young in dataset: \n'+classification_report(y_true_young, y_pred_young))
conf_mat_young = confusion_matrix(y_true_young, y_pred_young)
#print('confusiton matrix: \n', conf_mat_young)
plt.figure()
df_cm = pd.DataFrame(conf_mat_young, categories, categories)
sns_plot = sns.heatmap(df_cm, annot=True)
sns_plot.figure.savefig(fig_path+'conf_mat_young.png')

## old
data_test_old = data_test.loc[data_test['Age'] == 'old']
y_pred_old = data_test_old['Predicted Label']
y_true_old = data_test_old['True Label']

h.write('Classification Report for old in dataset: \n'+ classification_report(y_true_old, y_pred_old))
conf_mat_old = confusion_matrix(y_true_old, y_pred_old)
#print('confusiton matrix: \n', conf_mat_old)
plt.figure()
df_cm = pd.DataFrame(conf_mat_old, categories, categories)
sns_plot = sns.heatmap(df_cm, annot=True)
sns_plot.figure.savefig(fig_path+'conf_mat_old.png')


h.close()