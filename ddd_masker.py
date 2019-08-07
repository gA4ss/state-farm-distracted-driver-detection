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
import pandas as pd

data_dir = 'D:/workspace/udacity/MLND/capstone/distracted_driver_detection/data/'

ON_CLIENT = True
NUM_CLASSES = 10
DROPOUT = 0.5
DEVICE_ID = 0
CROP_SIZE = (320, 480)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
IMG_MEAN = (0.3184719383716583, 0.3813590109348297, 0.37875279784202576)
IMG_VAR = (0.3184719383716583, 0.3813589811325073, 0.37875282764434814)
IMG_STD = (0.5643, 0.6175, 0.6154)
KEEP_LAYER = None

CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if CUDA else "cpu")

DEVICE_MASKER = torch.device("cpu")


def device_setting(on_client):
    if on_client is True:
        DEVICE_MASKER = torch.device("cuda:0")
    else:
        DEVICE_MASKER = torch.device("cuda:3")


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
        super().__init__(make_layers(cfg[model]))
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
                 keep_layer=KEEP_LAYER):
        super(MaskNet, self).__init__()

        if model is None:
            self.featurer = VGGNet()
        else:
            self.featurer = model

        self.dropout = dropout
        self.num_classes = num_classes
        self.keep_layer = keep_layer

        for name, param in self.featurer.named_parameters():
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
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.masker = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )

        self.make_classifier('xception', 'imagenet')

    def extract_mask_image(self, x, show=False, show_index=0):
        feature_image = self.extract_feature_image(
            x, show=show, show_index=show_index)
        mask = self.make_mask(feature_image, show=show, show_index=show_index)
        mask_feature_image = self.make_image(
            x, mask, show=show, show_index=show_index)
        return mask_feature_image

    def extract_feature_image(self, x, show=False, show_index=0):
        output = self.featurer(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        feature = self.bn1(self.relu(self.deconv1(x5)))
        feature = feature + x4
        feature = self.bn2(self.relu(self.deconv2(feature)))
        feature = feature + x3
        feature = self.bn3(self.relu(self.deconv3(feature)))
        feature = feature + x2
        feature = self.bn4(self.relu(self.deconv4(feature)))
        feature = feature + x1
        feature = self.bn5(self.relu(self.deconv5(feature)))

        if show is True:
            image = transforms.functional.to_pil_image(
                feature[show_index].cpu().detach())
            image.show()

        return feature

    def make_mask(self, feature, show=False, show_index=0):
        mask = self.masker(feature)
        self.tensor_filter(mask)
        mask[mask > mask.mean()] = 0
        mask[mask < mask.mean()] = 1

        if show is True:
            image = transforms.functional.to_pil_image(
                mask[show_index].cpu().detach())
            image.show()

        return mask

    def make_image(self, x, mask, show=False, show_index=0):
        # show image after masked
        image = x * mask
        #show = transforms.functional.to_pil_image(image[0].cpu().detach())
        # show.show()

        if show is True:
            show_image = transforms.functional.to_pil_image(
                image[show_index].cpu().detach())
            show_image.show()

        return image

    def forward(self, x):
        #x = self.extract_feature_image(x)
        pred = self.classifier(x)
        return pred

    def make_classifier(self, model_name='xception', pretrained='imagenet'):
        self.classifier = pretrainedmodels.__dict__[
            model_name](pretrained=pretrained)

        for param in self.classifier.parameters():
            param.requires_grad = False

        in_dim = self.classifier.last_linear.in_features

        self.classifier_last_linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(in_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, self.num_classes)
        )

        self.classifier.last_linear = self.classifier_last_linear

    def ModePool2d(self, input_tensor, kernel_size=3):
        row_size = input_tensor.size(0)
        col_size = input_tensor.size(1)

        row_count = row_size // kernel_size
        col_count = col_size // kernel_size

        for i in range(0, row_count):
            for j in range(0, col_count):
                m = input_tensor[i*kernel_size: i*kernel_size + kernel_size,
                                 j*kernel_size: j*kernel_size+kernel_size]
                feature_value = m.mode()[0].mode()[0]
                input_tensor[i*kernel_size: i*kernel_size + kernel_size,
                             j*kernel_size: j*kernel_size+kernel_size] = feature_value
        return input_tensor

    def tensor_filter(self, tensors_items):
        for item in tensors_items:
            for tensor in item:
                self.ModePool2d(tensor)

    def check_keep_layer(self, name):
        if self.keep_layer is None:
            return False

        if len(self.keep_layer) == 0:
            return False

        for kl in self.keep_layer:
            if kl in name:
                return True
        return False


class DriverDataset(Dataset):
    def __init__(self, data_dir, data_frame, subject):
        super(DriverDataset, self).__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.df = data_frame
        self.count = 0

        self.transorms_gen = transforms.Compose([
            transforms.CenterCrop(CROP_SIZE),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

        self.image_paths = {}
        self.image_paths_ = []
        for s in range(0, 10):
            state = 'c{}'.format(s)
            self.image_paths[state] = []
            items = self.df[(self.df["subject"] == subject)]
            for _, row in items.iterrows():
                subpath = row["classname"] + "/" + row["img"]
                path = os.path.join(self.data_dir, subpath)
                self.image_paths[state].append(path)
                path_state = (path, s)
                self.image_paths_.append(path_state)
                self.count += 1

        print("{} number of {}'s image loaded.".format(self.count, subject))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        # assert index >= len(self.image_paths), 'index, out of range'
        path = self.image_paths_[index][0]
        image = Image.open(path)
        filename = os.path.basename(path)
        # image_tensor = transforms.functional.to_tensor(image)
        label = self.image_paths_[index][1]
        transorms_image = self.transorms_gen(image)
        return (transorms_image, (label, filename))


class ClassDataset(Dataset):
    def __init__(self, data_dir, sub_dir='train'):
        super(ClassDataset, self).__init__()
        read_dir = os.path.join(data_dir, sub_dir)
        self.data_dir = os.path.abspath(read_dir)
        self.test = True if sub_dir == 'test' else False

        self.transorms_gen = transforms.Compose([
            transforms.CenterCrop(CROP_SIZE),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

        self.image_paths = {}
        self.image_paths_ = []

        if sub_dir == 'test':
            self.image_paths_ = glob(os.path.join(self.data_dir, '*.jpg'))
            self.image_paths = self.image_paths_
        else:
            for s in range(0, 10):
                state = 'c{}'.format(s)
                read_path = os.path.join(self.data_dir, state)
                self.image_paths[state] = glob(
                    os.path.join(read_path, '*.jpg'))
                path_states = [(path, s) for path in self.image_paths[state]]
                self.image_paths_.extend(path_states)

        print("{} number of image loaded.".format(len(self.image_paths_)))

    def __len__(self):
        return len(self.image_paths_)

    def __getitem__(self, index):
        # assert index >= len(self.image_paths), 'index, out of range'
        if self.test is False:
            path = self.image_paths_[index][0]
            image = Image.open(path)
            filename = os.path.basename(path)
            # image_tensor = transforms.functional.to_tensor(image)
            label = self.image_paths_[index][1]
            transorms_image = self.transorms_gen(image)
        else:
            path = self.image_paths_[index]
            image = Image.open(path)
            filename = os.path.basename(path)
            label = 0
            transorms_image = self.transorms_gen(image)
        return (transorms_image, (label, filename))


def generate_mask_image(data_dir='../data/', save_dir='../data/mask/',
                        sub_dir='train', batch_size=64, model_path=None,
                        on_client=ON_CLIENT):
    device_setting(on_client)
    data_dir = os.path.abspath(data_dir)
    if model_path is not None:
        model_path = os.path.abspath(model_path)
    save_dir = os.path.abspath(save_dir)

    masker_model = MaskNet()
    masker_model.to(DEVICE_MASKER)

    if model_path is not None:
        masker_model.load_state_dict(torch.load(
            model_path, map_location=DEVICE_MASKER))
    print('[INFO]load model ok')

    # mask_gen = ImageFolder(os.path.join(
    #    data_dir, sub_dir), transform=transorms_gen)
    image_dataset = ClassDataset(data_dir=data_dir, sub_dir=sub_dir)
    mask_dataiter = DataLoader(
        image_dataset, batch_size=batch_size, shuffle=False)
    print('[INFO]generate datasets ok')

    output_dir = os.path.join(save_dir, sub_dir)
    masker_model.eval()
    for i, data in enumerate(mask_dataiter):
        print('Batch {}'.format(i))
        images = data[0]
        labels = data[1][0]
        filenames = data[1][1]

        images = images.to(DEVICE_MASKER)
        masked_images = masker_model.extract_mask_image(images, show=False)
        for i, masked_image in enumerate(masked_images):
            filename = filenames[i]
            output_name = 'masked_{}'.format(filename)
            if sub_dir == 'test':
                output_path = os.path.join(output_dir, output_name)
            else:
                label = labels[i]
                output_path = os.path.join(output_dir, 'c{}'.format(label))
            output_image = transforms.functional.to_pil_image(
                masked_image.cpu())
            output_image.save(output_path)
            print('save {} to {}'.format(output_name, output_path))


# ========================================================================================================
#generate_mask_image(data_dir=data_dir+'../data/', batch_size=1, on_client=True)
if __name__ == '__main__':
    fire.Fire()
