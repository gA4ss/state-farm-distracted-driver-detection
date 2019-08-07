# coding:utf-8
"""
"""
import os
import sys
import cv2
import csv
import fire
import time
import json
import tarfile
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pretrainedmodels

from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

data_dir = 'D:/workspace/udacity/MLND/capstone/distracted_driver_detection/data/'

ON_CLIENT = True
DEVICE_ID = 0
EPOCHS = 10
DROPOUT = 0.5
NUM_CLASSES = 10
LEARN_RATE = 0.001
CROP_SIZE = (320, 480)
IMG_SIZE = (224, 224)
BATCH_SIZE = 1
IMG_MEAN = (0.3184719383716583, 0.3813590109348297, 0.37875279784202576)
IMG_VAR = (0.3184719383716583, 0.3813589811325073, 0.37875282764434814)
IMG_STD = (0.5643, 0.6175, 0.6154)
KEEP_LAYER = None
FAKE_TRAIN = False

CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if CUDA else "cpu")

DEVICE_MASKER = torch.device("cuda:1")

def device_setting(on_client):
    if on_client is True:
        DEVICE_MASKER = torch.device("cuda:0")
    else:
        DEVICE_MASKER = torch.device("cuda:1")

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16'):
        super().__init__(make_layers(cfg[model], batch_norm=False))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, (begin, end) in enumerate(self.ranges):
            # self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d" % (idx+1)] = x

        return output


class MaskNet(nn.Module):
    def __init__(self, model=None, dropout=DROPOUT, num_classes=NUM_CLASSES,
                 keep_layer=KEEP_LAYER, tensor_filter_cb=None):
        super(MaskNet, self).__init__()

        if model is None:
            self.model = VGGNet()
        else:
            self.model = model

        self.vgg16 = models.vgg16(pretrained=True)
        self.xception = pretrainedmodels.__dict__['xception'](pretrained='imagenet')

        self.dropout = dropout
        self.num_classes = num_classes
        self.keep_layer = keep_layer

        for name, param in self.xception.named_parameters():
            param.requires_grad = False

        for name, param in self.vgg16.named_parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            param.requires_grad = self.check_keep_layer(name)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 3, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(3)

        #self.masker = nn.Conv2d(32, 3, kernel_size=3, stride=1 ,padding=1)
        #self.excessive = nn.Conv2d(32, 1, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(1000, self.num_classes)
        )

    def forward(self, x):
        output = self.model(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        mask = self.bn1(self.relu(self.deconv1(x5)))
        mask = mask + x4
        mask = self.bn2(self.relu(self.deconv2(mask)))
        mask = mask + x3
        mask = self.bn3(self.relu(self.deconv3(mask)))
        mask = mask + x2
        mask = self.bn4(self.relu(self.deconv4(mask)))
        mask = mask + x1
        mask = self.bn5(self.relu(self.deconv5(mask)))

        #self.deconv_output = mask
        #output = torch.zeros(IMG_SIZE, device=DEVICE_MASKER)
        #for m in mask[0]:
        #    output += m
        #output = mask[0][0]
        #output_img = transforms.functional.to_pil_image(output.detach().cpu())
        #output_img.show()
        #self.mask = self.masker(mask)

        # use filter to mask
        # self.tensor_filter(self.mask)

        # mask_cpu = self.mask.cpu()
        # mask = transforms.functional.to_pil_image(mask_cpu)
        # img.show()

        #self.feature_image = self.excessive(self.mask)
        self.feature_image = None
        x = self.xception(mask)
        #x = self.feature_image.view(self.feature_image.size(0), -1)
        pred = self.classifier(x)

        return pred

    def tensor_filter(self, tensors_items):
        for item in tensors_items:
            for tensor in item:
                mean = tensor.mean()
                tensor[tensor < mean] = 0

    def check_keep_layer(self, name):
        if self.keep_layer is None:
            return False

        if len(self.keep_layer) == 0:
            return False

        for kl in self.keep_layer:
            if kl in name:
                return True
        return False


# ========================================================================================================

def model_print():

    model = MaskNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    with SummaryWriter(log_dir='../ddd_model', comment='Masknet')as w:
        w.add_graph(model, dummy_input)
    
if __name__ == '__main__':
    model_print()
