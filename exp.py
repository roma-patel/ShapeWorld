#!/usr/bin/env python
import os
import re
import torch
import torch.nn as nn
from models.alexnet import alexnet
from models.resnet import resnet101
from models.inception import inception_v3
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_resnet101():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # downloads pretrained resnet-101
    model = resnet101(pretrained=True)
    print model.fc
    traindir = os.path.join(os.getcwd(), 'data/train/')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    print train_dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset)

    '''
Inside forward!
(1, 3, 224, 224)
(1, 64, 112, 112)
(1, 64, 112, 112)
(1, 64, 112, 112)
(1, 64, 56, 56)
(1, 256, 56, 56)
(1, 512, 28, 28)
(1, 1024, 14, 14)
(1, 2048, 7, 7)
(1, 2048, 1, 1)
(1, 2048)
(1, 1000)
    '''
    for i, (input, target) in enumerate(train_loader):
        '''
        print 'i: ' + str(i)
        print 'input: ' + str(input)
        print 'target: ' + str(target)
        print '\n\n'
        '''
        x = model.forward(input)
        print 'iter: ' + str(i)
        print x.size()


def load_alexnet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # downloads pretrained alexnet
    model = alexnet(pretrained=True)
    # loads new images from traindir
    traindir = os.path.join(os.getcwd(), 'data/train/')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    print train_dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset)

    # alexnet but conflated into features with Sequential

    '''
Inside forward!
(1, 3, 224, 224)
(1, 256, 6, 6)
(1, 9216)
(1, 1000) <-- after self.fc
    '''
    for i, (input, target) in enumerate(train_loader):
        '''
        print 'i: ' + str(i)
        print 'input: ' + str(input)
        print 'target: ' + str(target)
        print '\n\n'
        '''
        x = model.forward(input)
        print 'iter: ' + str(i)
        print x.size()


def load_inception():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # downloads pretrained inception
    model = inception_v3(pretrained=True)
    # loads new images from traindir
    traindir = os.path.join(os.getcwd(), 'data/train/')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    print train_dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset)

    '''
Inside forward!
(1, 3, 224, 224)
(1, 32, 111, 111)
(1, 32, 109, 109)
(1, 64, 109, 109)
(1, 64, 54, 54)
(1, 80, 54, 54)
(1, 192, 52, 52)
(1, 192, 25, 25)
(1, 256, 25, 25)
(1, 288, 25, 25)
(1, 288, 25, 25)
(1, 768, 12, 12)
(1, 768, 12, 12)
(1, 768, 12, 12)
(1, 768, 12, 12)
(1, 768, 12, 12)
(1, 1280, 5, 5)
(1, 2048, 5, 5)
(1, 2048, 5, 5)
(1, 2048, 1, 1)
(1, 2048, 1, 1)
(1, 2048)
(1, 1000) <-- after self.fc
    '''
    for i, (input, target) in enumerate(train_loader):
        '''
        print 'i: ' + str(i)
        print 'input: ' + str(input)
        print 'target: ' + str(target)
        print '\n\n'
        '''
        x = model.forward(input)
        print 'iter: ' + str(i)
        print x.size()
        # convert tensor (grad) to array and save features
        
if __name__ == '__main__':
    #load_resnet101()
    #load_alexnet()
    load_inception()
