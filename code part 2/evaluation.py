import pandas as pd
import numpy as np
import pickle
import os
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


"""## Loading Data

In this loading phase of our project we are labeling data as well as images into a list
"""

DIRECTORY = 'Data/'
CATEGORIES = ['Cloth mask','Mask worn incorrectly','N-95_Mask','No Face Mask','Surgical Mask']

data = []
labels = []
 
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    average1 = 0  
    average2 = 0
    number =  0
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        
        try:
            image = Image.open(img_path)
            average1 +=image.size[0]
            average2 +=image.size[1]
            number +=1
            data.append(image)
            labels.append(category)
        except:
            pass  
    average1 = average1 /number 
    average2 = average2 /number 
    
    # print("Average for class ",category," is ", average1, "X", average2)

"""## Split dataset into train, validation, and test datasets

This is spliting phase. For this task, sklearn has been used. 
"""

total_images = len(data)
train_dataset_size = 1590
test_dataset_size = total_images - train_dataset_size
test_dataset_percentage = test_dataset_size/total_images
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_dataset_percentage, random_state=42)
data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=0.1, random_state=42)

# print('Before Preprocessing ...')
# print('Number of Train Dataset Images:', len(data_train))
# print('Number of Validation Dataset Images:', len(data_val))
# print('Number of Test Dataset Images:', len(data_test))

"""## Preprocessing

All images are in different shape and size so before feeding to CNN network. All of them must be converted in same size (hight, width, and depth). Also for better results from model, a few images have been fliped or rotated. 
"""

# Imagenet standards
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Train uses data augmentation
train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))  
])
    
# Validation does not use augmentation
test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

"""For each class, a label in integer format is assigned:  <br>
0: Cloth Mask, 1: Mask Worn Incorrectly, 2: N-95 Mask, 3: NO Face Mask 4: Surgical Mask
"""

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
labels_train = lb_make.fit_transform(labels_train)
labels_val = lb_make.transform(labels_val)
labels_test = lb_make.transform(labels_test)

train_images = []
train_labels = []


for i in range(len(data_train)):
    try:
        train_images.append(train_transforms(data_train[i]))
        train_labels.append(labels_train[i])
    except:
        pass
    
val_images = []
val_labels = []
for i in range(len(data_test)):
    try:
        val_images.append(test_transforms(data_val[i]))
        val_labels.append(labels_val[i])
    except:
        pass

test_images = []
test_labels = []
for i in range(len(data_test)):
    try:
        test_images.append(test_transforms(data_test[i]))
        test_labels.append(labels_test[i])
    except:
        pass

"""Here, all numpy arrays are converted into tensor."""

train_images_numpy = [t.numpy() for t in train_images]
val_images_numpy = [t.numpy() for t in val_images]
test_images_numpy = [t.numpy() for t in test_images]

train_images_tensor = torch.tensor(train_images_numpy)
val_images_tensor = torch.tensor(val_images_numpy)
test_images_tensor = torch.tensor(test_images_numpy)

train_labels_tensor = torch.tensor(train_labels)
val_labels_tensor = torch.tensor(val_labels, dtype = torch.long)
test_labels_tensor = torch.tensor(test_labels)

# torch.save(test_images_tensor,'test_images_tensor.pth')

# print('After Preprocessing ...')
# print('Number of Train Dataset Images:', len(train_images_tensor))
# print('Number of Validation Dataset Images:', len(val_images_tensor))
# print('Number of Test Dataset Images:', len(test_images_tensor))
# print('Total Number of Training Images:', len(train_images_tensor)+len(val_images_tensor))
# print('Total Number of Test Images:', len(test_images_tensor))
# print('Total Number of Images:', len(train_images_tensor)+len(val_images_tensor)+ len(test_images_tensor))

"""## Importing data into Pytroch DataLoader"""

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
val_dataset = CreateDataset(images=val_images_tensor, labels=val_labels_tensor)
test_dataset = CreateDataset(images=test_images_tensor, labels=test_labels_tensor)

batch_size_train = 64
batch_size_val = 64
batch_size_test = 64

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size_train,
                                           shuffle=True,
                                           drop_last=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size_val,
                                           shuffle=True,
                                           drop_last=False)                                           

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                         batch_size=batch_size_test,
                                         shuffle=False,
                                         drop_last=False)

#torch.save(test_loader, 'test_loader.pth')

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break
        
show_batch(train_loader)
plt.savefig('Figures/images_after_pre_process.jpg')


"""## CreateDataset"""

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

"""## Models"""

## Model 1



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

## Model 2

class net_ver2(nn.Module):
    def __init__(self):
        super(net_ver2, self).__init__()
        
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
        
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(56 * 56 * 32, 128),
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
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return F.log_softmax(x, dim = 1)

## Model 3

class net_ver3(nn.Module):
    def __init__(self):
        super(net_ver3, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(224 * 224 * 32, 128),
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
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim = 1)


"""Test function"""

def test(model, test_loader, num_test_images):

    # Test:
    output_true = []
    output_pred = []
    batch_size_test = 64

    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images,labels) in enumerate(test_loader):
            output_true.append(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            output_pred.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = (correct / total) * 100

        print('Testing Accuracy of the model on the test images: {} %'
            .format(test_acc))

    y_pred = np.zeros(num_test_images)
    y_true = np.zeros(num_test_images)
    for i in range(len(output_pred)):
        for j in range(output_pred[i].shape[0]):
            y_pred[i*(batch_size_test)+j] = output_pred[i][j]
            y_true[i*(batch_size_test)+j] = output_true[i][j]

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    return model, y_pred, y_true

"""## Test Models
Model 1
Download model from this link -  https://drive.google.com/file/d/1-lxymdGwY50GvSck-A2WFhpD5lSwSM8Q/view?usp=sharing 
put it inside Models/ver1/
Then, run the below code. 
"""
## Load Test Loader
#test_loader = torch.load('test_loader.pth')
#test_images_tensor = torch.load('test_images_tensor.pth')

load_model = True
path_model = 'Models/ver1/model_100.pth'
path_dict = 'Models/ver1/info_loss_acc.pkl'
if load_model == True:
    model_1 = torch.load(path_model)
    model_1.eval()
    with open(path_dict, 'rb') as f:
        info_loss_acc_1 = pickle.load(f)
num_test_images = len(test_images_tensor)
model_1, y_pred_1, y_true_1 = test(model_1, test_loader, num_test_images)

loss_train_avg = info_loss_acc_1['train_loss']
acc_train_avg = info_loss_acc_1['train_acc ']
loss_val_avg = info_loss_acc_1['val_loss'] 
acc_val_avg = info_loss_acc_1['val_acc']

## Loss
plt.figure(figsize=(10,7))
plt.plot(loss_train_avg, 'c', markersize=12, markeredgewidth=3
         , linewidth=2,label='training_loss')
plt.plot(loss_val_avg, 'b', markersize=12, markeredgewidth=3
         , linewidth=2,label='val_loss')
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig('Figures/ver1/loss.jpg')

## Accuracy
plt.figure(figsize=(10,7))
plt.plot(acc_train_avg, 'c', markersize=12, markeredgewidth=3
         , linewidth=2,label='training_acc')
plt.plot(acc_val_avg, 'b', markersize=8, markeredgewidth=3
         , linewidth=3,label='val_acc')
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.savefig('Figures/ver1/accuracy.jpg')

print('Classification Report: \n', classification_report(y_true_1, y_pred_1))
conf_mat = confusion_matrix(y_true_1, y_pred_1)
print('confusiton matrix: \n', conf_mat)

plt.figure()
categories = ['Cloth','Incorrectly', 'N95', 'NoMask', 'Surgical']
df_cm = pd.DataFrame(conf_mat, categories, categories)
sns_plot = sns.heatmap(df_cm, annot=True)
sns_plot.figure.savefig("Figures/ver1/conf_mat.png")

"""
Model 2
Download model from this link - https://drive.google.com/file/d/1-lxymdGwY50GvSck-A2WFhpD5lSwSM8Q/view?usp=sharing
put it inside Models/ver2/
Then, run the below code. 
"""

load_model = True
path_model = 'Models/ver2/model_100.pth'
path_dict = 'Models/ver2/info_loss_acc.pkl'
if load_model == True:
    model_2 = torch.load(path_model)
    model_2.eval()
    with open(path_dict, 'rb') as f:
        info_loss_acc_2 = pickle.load(f)
num_test_images = len(test_images_tensor)
model_2, y_pred_2, y_true_2 = test(model_2, test_loader, num_test_images)

loss_train_avg = info_loss_acc_2['train_loss']
acc_train_avg = info_loss_acc_2['train_acc ']
loss_val_avg = info_loss_acc_2['val_loss'] 
acc_val_avg = info_loss_acc_2['val_acc']

## Loss
plt.figure(figsize=(10,7))
plt.plot(loss_train_avg, 'c', markersize=12, markeredgewidth=3
         , linewidth=2,label='training_loss')
plt.plot(loss_val_avg, 'b', markersize=12, markeredgewidth=3
         , linewidth=2,label='val_loss')
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig('Figures/ver2/loss.jpg')

## Accuracy
plt.figure(figsize=(10,7))
plt.plot(acc_train_avg, 'c', markersize=12, markeredgewidth=3
         , linewidth=2,label='training_acc')
plt.plot(acc_val_avg, 'b', markersize=8, markeredgewidth=3
         , linewidth=3,label='val_acc')
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.savefig('Figures/ver2/accuracy.jpg')

print('Classification Report: \n', classification_report(y_true_2, y_pred_2))
conf_mat = confusion_matrix(y_true_2, y_pred_2)
print('confusiton matrix: \n', conf_mat)

plt.figure()
categories = ['Cloth','Incorrectly', 'N95', 'NoMask', 'Surgical']
df_cm = pd.DataFrame(conf_mat, categories, categories)
sns_plot = sns.heatmap(df_cm, annot=True)
sns_plot.figure.savefig("Figures/ver2/conf_mat.png")

"""#### Model 3
Download model from this link - https://drive.google.com/file/d/1ebwzJUXSjOn7HY8MIWp2i56SdZAFvTfG/view?usp=sharing
put it inside Models/ver3/
Then, run the below code. 
"""

load_model = True
path_model = 'Models/ver3/model_100.pth'
path_dict = 'Models/ver3/info_loss_acc.pkl'
if load_model == True:
    model_3 = torch.load(path_model)
    model_3.eval()
    with open(path_dict, 'rb') as f:
        info_loss_acc_3 = pickle.load(f)
num_test_images = len(test_images_tensor)
model_3, y_pred_3, y_true_3 = test(model_3, test_loader, num_test_images)

loss_train_avg = info_loss_acc_3['train_loss']
acc_train_avg = info_loss_acc_3['train_acc ']
loss_val_avg = info_loss_acc_3['val_loss'] 
acc_val_avg = info_loss_acc_3['val_acc']

## Loss
plt.figure(figsize=(10,7))
plt.plot(loss_train_avg, 'c', markersize=12, markeredgewidth=3
         , linewidth=2,label='training_loss')
plt.plot(loss_val_avg, 'b', markersize=12, markeredgewidth=3
         , linewidth=2,label='val_loss')
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig('Figures/ver3/loss.jpg')

## Accuracy
plt.figure(figsize=(10,7))
plt.plot(acc_train_avg, 'c', markersize=12, markeredgewidth=3
         , linewidth=2,label='training_acc')
plt.plot(acc_val_avg, 'b', markersize=8, markeredgewidth=3
         , linewidth=3,label='val_acc')
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.savefig('Figures/ver3/accuracy.jpg')


print('Classification Report: \n', classification_report(y_true_3, y_pred_3))
conf_mat = confusion_matrix(y_true_3, y_pred_3)
print('confusiton matrix: \n', conf_mat)

plt.figure()
categories = ['Cloth','Incorrectly', 'N95', 'NoMask', 'Surgical']
df_cm = pd.DataFrame(conf_mat, categories, categories)
sns_plot = sns.heatmap(df_cm, annot=True)
sns_plot.figure.savefig("Figures/ver3/conf_mat.png")
