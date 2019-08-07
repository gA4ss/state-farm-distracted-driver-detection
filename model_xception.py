# coding:utf-8
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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pretrainedmodels

from glob import glob
from PIL import Image
from datetime import datetime
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter

ON_CLIENT = True
NUM_CLASSES = 10
EPOCHS = 10
LEARN_RATE = 0.001
DROPOUT = 0.5
DEVICE_ID = 0
CROP_SIZE = (320, 480)
#MG_SIZE = (224, 224)
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
IMG_MEAN = (0.31633861, 0.38164441, 0.37510719)
IMG_STD = (0.28836174, 0.32873901, 0.33058995)
DEFAULT_DEVICE_ID = 0
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cpu")

data_dir = 'D:/workspace/udacity/MLND/capstone/distracted_driver_detection/data/'


class XceptionNet(nn.Module):
    def __init__(self, model=None, dropout=DROPOUT, num_classes=NUM_CLASSES):
        super(XceptionNet, self).__init__()

        if model is None:
            self.model = pretrainedmodels.__dict__[
                'xception'](pretrained='imagenet')
        else:
            self.model = model
        # self.model.to(DEVICE_CLASSIFIER)

        self.dropout = dropout
        self.num_classes = num_classes

        # self.classifier_transorms = transforms.Compose([
        #    transforms.Lambda(lambda tensors:
        #                      [transforms.Resize((299, 299))(transforms.ToPILImage()(tensor)) for tensor in tensors]),
        #    transforms.Lambda(lambda tensors:
        #                      torch.stack([transforms.ToTensor()(tensor) for tensor in tensors]))
        # ])

        # extract freature
        # self.feature = nn.Sequential(*list(self.model.children())[:-1])

        # for param in self.model.parameters():
        #    param.requires_grad = False

        in_dim = self.model.last_linear.in_features
        last_linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(in_dim, self.num_classes)
        )

        self.model.last_linear = last_linear

    def forward(self, x):
        x = self.model(x)
        return x


def make_model(model_path=None, dropout=DROPOUT, 
               num_classes=NUM_CLASSES, map_location=DEVICE,
               on_client=ON_CLIENT):
    model = XceptionNet(
        dropout=dropout, num_classes=num_classes)
    if model_path is not None:
        print('[INFO]map model to {} device.'.format(str(map_location)))
        model.load_state_dict(torch.load(
            model_path, map_location=map_location))
    else:
        model = model.to(map_location)

    return model
