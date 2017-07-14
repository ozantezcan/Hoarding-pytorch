from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchsample
from torchsample import transforms as ts_transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import Image

from torchsample.transforms import RangeNorm

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                            
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight,weight_per_class

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, lr_scheduler,dset_loaders,dset_sizes,writer,use_gpu=True, num_epochs=25,batch_size=4,num_log=100):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                batch_count=0
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_cir1=0
            running_hist=np.zeros(9)

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                labels_multi=[]
                for label in labels.data:
                    label_multi=np.zeros(11)
                    label_multi[label:label+3]=1
                    label_multi=label_multi[1:-1]
                    labels_multi.append(label_multi)
                labelsv = Variable(torch.FloatTensor(labels_multi).cuda()).view(-1,9)

                loss = criterion(outputs, labelsv)

                # backward + optimize only if in training phase
                if phase == 'train':
                    batch_count+=1
                    if(np.mod(batch_count,num_log)==0):
                        batch_loss = running_loss / (batch_count*batch_size)
                        batch_acc = running_corrects / (batch_count*batch_size)
                        batch_cir1 = running_cir1 / (batch_count*batch_size)

                        print('{}/{}, CIR-1: {:.4f}'
                              .format(batch_count,len(dset_loaders['train']),batch_cir1))
                        
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                running_cir1 += torch.sum(torch.abs(preds - labels.data)<=1)
                for k in range(9):
                    if(torch.sum(labels.data==k)>0):
                        running_hist[k] += torch.sum(torch.abs(preds[labels.data==k] - k)<=1)/torch.sum(labels.data==k)
                
                

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            epoch_cir1 = running_cir1 / dset_sizes[phase]
            epoch_hist = running_hist
            writer.add_scalar(phase+' loss',epoch_loss,epoch)
            writer.add_scalar(phase+' accuracy',epoch_acc,epoch)
            writer.add_scalar(phase+' CIR-1',epoch_cir1,epoch)
            writer.add_histogram(phase+' Histogram',epoch_hist,epoch)


			
            print('{} Loss: {:.4f} Acc: {:.4f} CIR-1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_cir1))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

def train_model_balanced(model, criterion, optimizer, lr_scheduler,dset_loaders,use_gpu=True, num_epochs=25,num_train=100,num_test=10,batch_size=4):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
            # Iterate over data.
        batch_count=0
        #
        optimizer = lr_scheduler(optimizer, epoch)
        model.train(True)  # Set model to training mode
        for opt_iter in range(num_test):  
            running_loss = 0.0
            running_corrects = 0
            running_cir1=0

            for k in range(num_train):
                # get the inputs
                inputs, labels = next(iter(dset_loaders['train']))

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase           
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                running_cir1 += torch.sum(torch.abs(preds - labels.data)<=1)

            epoch_loss = running_loss / (num_train*batch_size)
            epoch_acc = running_corrects / (num_train*batch_size)
            epoch_cir1 = running_cir1 / (num_train*batch_size)

            print('{}/{}, CIR-1: {:.4f}'
                  .format(k,num_train,epoch_cir1))

            # deep copy the model
        model.train(False)  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        running_cir1=0
            
        for data in dset_loaders['val']:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                    Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
            running_cir1 += torch.sum(torch.abs(preds - labels.data)<=1)

        epoch_loss = running_loss / dset_sizes['val']
        epoch_acc = running_corrects / dset_sizes['val']
        epoch_cir1 = running_cir1 / dset_sizes['val']

        print('Val Loss: {:.4f} Acc: {:.4f} CIR-1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_cir1))


        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dset_loaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[labels.data[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return



