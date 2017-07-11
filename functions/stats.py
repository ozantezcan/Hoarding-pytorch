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


def im2torchNorm(imdir,mean = np.array([0.485, 0.456, 0.406]),std = np.array([0.229, 0.224, 0.225])\
                 ,imsize=(224,224),imMax=255.):
    im = Image.open(imdir)
    im.thumbnail(imsize, Image.ANTIALIAS) # resizes image in-place
    im=np.asarray(im).astype(np.float)/imMax
    im_norm=(im-mean)/std
    return im_norm

def subsetCreator(rootdir,im_per_room=10):
    subdirs=os.listdir(rootdir)
    roomdirs=['//BR//','//Kitchen//','//LR//']
    imdirs=[]
    cir=[]
    room=[]
    house=[]

    for hme in range(len(subdirs)):
        for rm in range(len(roomdirs)):
            for cr in range(9):
                parentdir=rootdir+subdirs[hme]+roomdirs[rm]+str(cr+1)
                imdirs_c=os.listdir(parentdir)
                if(len(imdirs_c)>0):
                    rand_idx=np.floor(np.random.rand(im_per_room)*len(imdirs_c)).astype(np.int)
                    for idx in rand_idx:
                        imdirs.append(parentdir+'//'+imdirs_c[idx])
                        house.append(hme+1)
                        room.append(rm+1)
                        cir.append(cr+1)
    return imdirs, cir, house, room

def torchFromDirs(imdirs,im_dims=[224,224,3],begin_idx=0,batch_size=16):
    if len(imdirs)<begin_idx:
        raise ValueError('Begin index cannot be higher than the length of image dirs')
    if len(imdirs)<begin_idx+batch_size:
        batch_size=len(imdirs)-begin_idx
    imgs=np.zeros([batch_size]+im_dims)
    for k in range(batch_size):
        imgs[k,:,:,:]=im2torchNorm(imdirs[begin_idx+k])

    im_torch=Variable(torch.from_numpy(imgs.transpose(0,3,1,2)).cuda()).float()
    return im_torch

def extractFeats(imdirs,network,batchsize=16,outsize=512):
    fvec=np.zeros([len(imdirs),outsize])

    for k in range(0,len(imdirs),batchsize):
        im_torch=torchFromDirs(imdirs,begin_idx=k,batch_size=batchsize)
        feat=network(im_torch)
        feat=feat.cpu()
        feat=feat.data.numpy()
        fvec[k:k+batchsize]=feat.reshape(-1,outsize)
    return fvec

def plotAll(fvec_tsne,cir,house,room,data_title=''):
    plt.figure()
    plt.scatter(fvec_tsne[:,0], fvec_tsne[:,1], c=cir, alpha=0.5)#,cmap='jet')
    plt.colorbar()
    plt.title(data_title+'Images by CIR')

    plt.figure()
    plt.scatter(fvec_tsne[:,0], fvec_tsne[:,1], c=house, alpha=0.5)#,cmap='jet')
    plt.colorbar()
    plt.title(data_title+'Images by House')

    plt.figure()
    plt.scatter(fvec_tsne[:,0], fvec_tsne[:,1], c=room, alpha=0.5)#,cmap='jet')
    plt.colorbar()
    plt.title(data_title+'Images by Room')

