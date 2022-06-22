"""
## Import Libraries

All all libraries must be installed which have been imported in the next code block.
"""

## Importing libraries
import os
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms


"""## Loading Data

In this loading phase of our project we are saving images 
and their corresponding labels into a list
"""

DIRECTORY = '../Data/Data part 2/'
CATEGORIES = ['Cloth mask','Mask worn incorrectly','N-95_Mask','No Face Mask','Surgical Mask']

data = []
name = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        
        try:
            image = Image.open(img_path)
            data.append(image)
            name.append(img)
            labels.append(category)
        except:
            pass  

"""## Split dataset into train, validation, and test datasets

This is spliting phase. For this task, sklearn has been used. 
"""

total_images = len(data)
train_dataset_size = 1570
test_dataset_size = total_images - train_dataset_size
test_dataset_percentage = test_dataset_size/total_images
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_dataset_percentage, random_state=42, stratify = labels)

print('Before Preprocessing ...')
print('Number of Train Dataset Images:', len(data_train))
print('Number of Test Dataset Images:', len(data_test))
print('Total Number of Images:', len(data_train)+ len(data_test))


"""## Preprocessing

All images are in different shape and size so before feeding to CNN network. All of them must be converted in same size (hight, width, and depth). Also for better results from model, a few images have been fliped or rotated. 
"""

saved_tensors_path = 'saved tensors/'

mean = torch.load(saved_tensors_path+'mean_train.pt')
std = torch.load(saved_tensors_path+'std_train.pt')

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
labels_test = lb_make.transform(labels_test)


train_images = []
train_labels = []


for i in range(len(data_train)):
    try:
        img_transform = train_transforms(data_train[i])
        if img_transform.shape == (3,224,224):
            train_images.append(img_transform)
            train_labels.append(labels_train[i])
    except:
        pass
    

test_images = []
test_labels = []
for i in range(len(data_test)):
    try:
        img_transform_test = train_transforms(data_test[i])
        if img_transform_test.shape == (3,224,224):
            test_images.append(img_transform_test)
            test_labels.append(labels_test[i])
    except:
        pass


"""Here, all numpy arrays are converted into tensor."""

train_images_numpy = [t.numpy() for t in train_images]
test_images_numpy = [t.numpy() for t in test_images]

train_images_tensor = torch.tensor(train_images_numpy)
test_images_tensor = torch.tensor(test_images_numpy)

train_labels_tensor = torch.tensor(train_labels)
test_labels_tensor = torch.tensor(test_labels)

print('After Preprocessing ...')
print('Number of Train Dataset Images:', len(train_images_tensor))
print('Number of Test Dataset Images:', len(test_images_tensor))
print('Total Number of Images:', len(train_images_tensor)+ len(test_images_tensor))

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


test_dataset = CreateDataset(images=test_images_tensor, labels=test_labels_tensor)

batch_size_test = 64
                                        
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size_test,
                                          shuffle=False,
                                          drop_last=False)


torch.save(train_images_tensor, saved_tensors_path+'train_images_tensor.pt')
torch.save(train_labels_tensor, saved_tensors_path+'train_labels_tensor.pt')
torch.save(test_loader, saved_tensors_path+'test_loader.pth')