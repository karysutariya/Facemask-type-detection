import os
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch.nn.functional as F


"""### Calculate mean and std of images"""


def mean_and_std(loader):
    mean = 0
    std= 0
    total_images_count = 0
    for images, _ in loader:
        images_count_in_a_batch = images.size(0)
        images = images.view(images_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images_count_in_a_batch
        
    mean /= total_images_count
    std /=  total_images_count
    
    return mean, std


"""### Train and Validation Function"""

def train_val(model_info, train_loader, val_loader, num_epochs, save_model = True):

    model = model_info['model']
    optimizer =  model_info['optimizer']
    criterion = model_info['criterion']
    path = model_info['path']

    loss_train = []
    loss_val = []
    loss_train_avg = []
    loss_val_avg = []

    acc_train = []
    acc_val = []
    acc_train_avg = []
    acc_val_avg = []

    info_loss_acc = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    # Training
    for epoch in range(num_epochs):
        
        model.train()
        loss_train = []
        acc_train = []
        
        for i, (images,labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_train.append(loss.item())
            
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_train.append(correct / total)

        loss_train_avg.append(sum(loss_train)/len(train_loader))
        acc_train_avg.append(sum(acc_train)/len(train_loader))
        
        # Validaiton
        loss_val = []
        acc_val = []
        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images,labels) in enumerate(val_loader):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_val.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc_val.append(correct / total)

            loss_val_avg.append(sum(loss_val)/len(val_loader))
            acc_val_avg.append(sum(acc_val)/len(val_loader))


        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
        .format(epoch + 1, num_epochs, loss_train_avg[epoch], acc_train_avg[epoch] * 100,
        loss_val_avg[epoch], acc_val_avg[epoch] * 100))
        
    if save_model:
        torch.save(model,path+'model_'+str(epoch+1)+'.pth')

    info_loss_acc['train_loss'] = loss_train_avg
    info_loss_acc['train_acc '] = acc_train_avg
    info_loss_acc['val_loss'] = loss_val_avg
    info_loss_acc['val_acc'] = acc_val_avg

    with open(path+'info_loss_acc.pkl', 'wb') as f:
        pickle.dump(info_loss_acc, f)

    return model, info_loss_acc


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

    return model, y_pred, y_true, test_acc