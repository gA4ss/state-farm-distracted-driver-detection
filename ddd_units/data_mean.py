# coding:utf-8

import os
import sys
import torch
import torchvision
import pretrainedmodels
import matplotlib.pyplot as plt
import numpy as np

from PIL import ImageFile
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

data_dir = 'D:/workspace/udacity/MLND/capstone/distracted_driver_detection/data/'

ImageFile.LOAD_TRUNCATED_IMAGES = True

def calc_mean_std(image_path, image_size=(224, 224), batch_size=32):
    my_transforms = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor()
    ])

    training_data = datasets.ImageFolder(image_path, transform=my_transforms)
    my_data_loader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True)

    means, std_dev = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    num_images = 0

    for idx, (data, label) in enumerate(my_data_loader):
        lst_chan_0, lst_chan_1, lst_chan_2 = data[:,
                                                  0, :, :], data[:, 1, :, :], data[:, 2, :, :]
        means += np.array([torch.mean(lst_chan_0),
                           torch.mean(lst_chan_1), torch.mean(lst_chan_2)])
        std_dev += np.array([torch.std(lst_chan_0),
                             torch.std(lst_chan_1), torch.std(lst_chan_2)])
        num_images += 1

        print("Batch {} mean sum={}, std sum={}".format(idx, means, std_dev))

    print("mean value of the channels: {}".format(means/num_images))
    print("standard deviation of the channels: {}".format(std_dev/num_images))

calc_mean_std(image_path='../data/train/')