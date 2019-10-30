from __future__ import print_function, division

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import openpyxl
import torch.nn as nn

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

def train_model(model, optimizer, lr_scheduler, dset_loaders,\
dset_sizes,writer,use_gpu=True, num_epochs=25,batch_size=4,num_log=100,\
init_lr=0.001,lr_decay_epoch=7, lmbda=0):

    since = time.time()

    best_model = model
    best_cir1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                batch_count=0
                if lr_scheduler is not None:
                    optimizer = lr_scheduler(optimizer, epoch,init_lr=init_lr,lr_decay_epoch=lr_decay_epoch)
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
                
                loss_ccr = nn.CrossEntropyLoss()(outputs, labels)
                labels_multi=[]
                for label in labels.data:
                    label_multi=np.zeros(11)
                    label_multi[label:label+3]=1
                    label_multi=label_multi[1:-1]
                    labels_multi.append(label_multi)
                    
                labelsv = Variable(torch.FloatTensor(labels_multi).cuda()).view(-1,9)
                loss_ccr1 = nn.MultiLabelSoftMarginLoss()(outputs, labelsv)
                
                loss = ((1-lmbda)*loss_ccr) + (lmbda*loss_ccr1)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    batch_count+=1
                    if(np.mod(batch_count,num_log)==0):
                        #batch_loss = running_loss / (batch_count*batch_size)
                        batch_acc = running_corrects / (batch_count*batch_size)
                        batch_cir1 = running_cir1 / (batch_count*batch_size)
                        #print(running_cir1.cpu().numpy(), batch_count, batch_size)

                        print('{}/{}, CCR: {:.4f}, CCR-1: {:.4f}'
                              .format(batch_count,len(dset_loaders['train']),batch_acc,batch_cir1))
                        
                    loss.backward()
                    optimizer.step()

                # statistics
                #print(loss.data[0])
                #print(preds, labels.data)
                running_loss += loss.data[0].cpu().numpy()
                running_corrects += torch.sum(preds == labels.data).cpu().numpy()
                running_cir1 += torch.sum(torch.abs(preds - labels.data)<=1).cpu().numpy()
                for k in range(9):
                    if(torch.sum(labels.data==k)>0):
                        running_hist[k] += torch.sum(torch.abs(preds[labels.data==k] - k)<=1)/torch.sum(labels.data==k)

            epoch_loss = running_loss/dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            epoch_cir1 = running_cir1 / dset_sizes[phase]
            #epoch_hist = running_hist
            #writer.add_scalar(phase+' loss',epoch_loss,epoch)
            #writer.add_scalar(phase+' accuracy',epoch_acc,epoch)
            #writer.add_scalar(phase+' CIR-1',epoch_cir1,epoch)
            #writer.add_histogram(phase+' Histogram',epoch_hist,epoch)
            if phase == 'train':
                epoch_loss_tr = epoch_loss
                epoch_cir1_tr = epoch_cir1

            print('{} Loss: {:.4f} CCR: {:.4f} CCR-1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_cir1))

            # deep copy the model
            if phase == 'test' and epoch_cir1 > best_cir1:
                best_cir1 = epoch_cir1
                best_model = copy.deepcopy(model)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_cir1))
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



