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

from utils import mean_and_std



"""## Loading Data

In this loading phase of our project we are labeling data as well as images into a list
"""
DIRECTORY = '../Data/Data part 2/'
CATEGORIES = ['Cloth mask','Mask worn incorrectly','N-95_Mask','No Face Mask','Surgical Mask']

data = []
labels = []
 
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        
        try:
            image = Image.open(img_path)
            data.append(image)
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


"""
All images are in different shape and size so before calculating mean and std of the images, 
they must be converted to the same size (hight, width, and depth). 
"""


# Train uses data augmentation
train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
#        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))  
])
    
"""For each class, a label in integer format is assigned:  <br>
0: Cloth Mask, 1: Mask Worn Incorrectly, 2: N-95 Mask, 3: NO Face Mask 4: Surgical Mask
"""

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
labels_train = lb_make.fit_transform(labels_train)


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
    


"""Here, all numpy arrays are converted into tensor."""

train_images_numpy = [t.numpy() for t in train_images]
train_images_tensor = torch.tensor(train_images_numpy)
train_labels_tensor = torch.tensor(train_labels)


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


batch_size_train = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size_train,
                                           shuffle=True,
                                           drop_last=False)


mean_train, std_train = mean_and_std(train_loader)

saved_tensors_path = 'saved tensors/'
if not os.path.exists(saved_tensors_path):
  os.makedirs(saved_tensors_path)

torch.save(mean_train, saved_tensors_path+'mean_train.pt')
torch.save(std_train, saved_tensors_path+'std_train.pt')
print('mean of Train dataset images: ', mean_train)
print('standard deviation of Train dataset images: ', std_train)