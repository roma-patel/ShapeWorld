#!/usr/bin/env python
import os, re, torch
import torch.nn as nn
from models.alexnet import alexnet
from models.resnet import resnet101
from models.inception import inception_v3
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import urllib
from urllib import request
from shapeworld import Dataset

def trace():
    print('Tracing!')
    dataset = Dataset.create(dtype='agreement', name='existential')
    generated = dataset.generate(n=128, mode='train', include_model=True)
    print('\n'.join(dataset.to_surface(value_type='language', word_ids=generated['caption'][:5])))

def load_resnet101():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # downloads pretrained resnet-101
    model = resnet101(pretrained=True)
    traindir = os.path.join(os.getcwd(), 'data/train/')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset)

    for i, (input, target) in enumerate(train_loader):
        x = model.forward(input)
        feat = x.data.numpy(); feat = feat[0]

def build():
    vocab = ['brown', 'robotics', 'navigation', 'great', 'is', 'for', 'and', 'linguistics', 'cognition']
    seq2seq = Seq2SeqEncoder()
    rnn = RNNEncoder(vocab=vocab, text_field_embedder=None, num_highway_layers=0, phrase_layer=seq2seq)
    sent = 'linguistics is great for robotics'
    
if __name__ == '__main__':
    trace()
    #load_inception()
