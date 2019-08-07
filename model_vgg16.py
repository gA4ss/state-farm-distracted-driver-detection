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
from torchvision.models.vgg import vgg16
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pretrainedmodels

from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_dir = 'D:/workspace/udacity/MLND/capstone/distracted_driver_detection/data/'

ON_CLIENT = True
NUM_CLASSES = 10
EPOCHS = 10
LEARN_RATE = 0.001
DROPOUT = 0.5
DEVICE_ID = 0
CROP_SIZE = (320, 480)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
IMG_MEAN = (0.31633861, 0.38164441, 0.37510719)
IMG_STD = (0.28836174, 0.32873901, 0.33058995)
KEEP_LAYER = None
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cpu")

class Vgg16Net(nn.Module):
    def __init__(self, model=None, dropout=DROPOUT, num_classes=NUM_CLASSES):
        super(Vgg16Net, self).__init__()

        if model is None:
            self.model = vgg16(pretrained=True)
        else:
            self.model = model

        self.dropout = dropout
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes)
        )
        self.model.classifier = self.classifier

    def forward(self, x):
        #x = self.extract_feature_image(x)
        #x = self.featurer(x)
        #x = x.view(x.size(0), -1)
        #y = self.classifier(x)
        y = self.model(x)
        return y


def make_model(model_path=None, dropout=DROPOUT, 
               num_classes=NUM_CLASSES, map_location=DEVICE,
               on_client=ON_CLIENT):
    model = Vgg16Net(
        dropout=dropout, num_classes=num_classes)
    if model_path is not None:
        print('[INFO]map model to {} device.'.format(str(map_location)))
        model.load_state_dict(torch.load(
            model_path, map_location=map_location))
    else:
        model = model.to(map_location)

    return model
