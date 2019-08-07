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
from torchvision.models.vgg import VGG
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pretrainedmodels

from glob import glob
from PIL import Image
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
IMG_SIZE = (224, 224)
#IMG_SIZE = (299, 299)
BATCH_SIZE = 32
IMG_MEAN = (0.31633861, 0.38164441, 0.37510719)
IMG_STD = (0.28836174, 0.32873901, 0.33058995)
KEEP_LAYER = None
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cpu")

data_dir = 'D:/workspace/udacity/MLND/capstone/distracted_driver_detection/data/'

class BranchNet(nn.Module):
    def __init__(self, dropout=DROPOUT, num_classes=NUM_CLASSES):
        super(BranchNet, self).__init__()

        self.dropout = dropout
        self.num_classes = num_classes

        #for param in self.featurer.parameters():
        #    param.requires_grad = False

        self.featurers = []
        self.classifiers = []
        for i in range(self.num_classes):
            sub_featurer = self.make_sub_featurer()
            sub_featurer.to(DEVICE)
            self.featurers.append(sub_featurer)
            sub_classifier = self.make_classifier()
            sub_classifier.to(DEVICE)
            self.classifiers.append(sub_classifier)

        self.channel = nn.Conv2d(512, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.channel.to(DEVICE)
        self.common_feature = self.make_main_featurer()
        self.common_feature.to(DEVICE)

    def forward(self, x):
        preds = []

        feature = self.common_feature(x)
        feature = self.channel(feature)
        for i in range(self.num_classes):
            single_feature = feature[:,i,:,:]
            single_feature = single_feature.unsqueeze(1)
            single_feature = self.featurers[i](single_feature)
            single_feature = feature.view(single_feature.size(0), -1)
            pred = self.classifiers[i](single_feature)
            preds.append(pred)
        preds = torch.stack(preds, dim=1)
        return preds

    def make_main_featurer(self):
        features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return features

    def make_sub_featurer(self):
        features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return features

    def make_classifier(self):
        classifier_last_linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(1690, 1)
        )
        return classifier_last_linear
# ========================================================================================================

def device_setting(on_client):
    if on_client is True:
        DEVICE = torch.device("cuda:0")
        #DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda:2")
    return DEVICE

def make_transorms():
    train_transorms = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        #transforms.RandomAffine(degrees=10, translate=(0.1, 0.3)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(IMG_SIZE),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    valid_transorms = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])
    return train_transorms, valid_transorms

def make_model(model_path=None, dropout=DROPOUT, on_client=ON_CLIENT):
    model = BranchNet(
        dropout=dropout, num_classes=NUM_CLASSES)
    if model_path is not None:
        model.load_state_dict(torch.load(
            model_path, map_location=DEVICE))

    return model


def get_datasets(data_dir, train_transorms, valid_transorms, batch_size=BATCH_SIZE):
    train_gen = ImageFolder(os.path.join(
        data_dir, 'train/'), transform=train_transorms)
    if valid_transorms is not None:
        valid_gen = ImageFolder(os.path.join(
            data_dir, 'valid/'), transform=valid_transorms)
    train_dataiter = DataLoader(train_gen, batch_size=batch_size, shuffle=True)
    return train_dataiter, valid_gen

def valid(model, criterion, valid_dataset, device=DEVICE):
    model.eval()

    loss = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(valid_dataset):
            image = data[0]
            label = torch.tensor(data[1], dtype=torch.long)

            image = torch.unsqueeze(image, 0)
            label = torch.unsqueeze(label, 0)

            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            loss += criterion(pred, label).item()
            pred = pred.max(1, keepdim=True)[1]
            correct += pred.eq(
                label.view_as(pred)).sum().item()
    loss /= len(valid_dataset)
    return loss, correct


def train_nn(model, criterion, optimizer, train_dataiter, device=DEVICE):
    model.train()

    for batch_index, train_data in enumerate(train_dataiter):
        images = train_data[0]
        labels = train_data[1]

        # copy CPU data to GPU
        images = images.to(device)
        labels = labels.to(device)
        #image = torch.unsqueeze(image.to(DEVICE), 0)
        #label = torch.unsqueeze(label.to(DEVICE), 0)

        optimizer.zero_grad()
        pred = model(images)

        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
    return loss

def train(model, data_dir, epochs=EPOCHS, batch_size=BATCH_SIZE,
          learn_rate=LEARN_RATE, dropout=DROPOUT, save_dir='../model', show_model=False,
          describe=None, zipit=False, on_client=ON_CLIENT, device=DEVICE):
    """
    trainning...
    """
    screen_print = []
    
    def printf(str):
        if str is None:
            print()
            screen_print.append('\n')
        else:
            print(str)
            screen_print.append(str+'\n')

    data_dir = os.path.abspath(data_dir)
    save_dir = os.path.abspath(save_dir)

    if describe is not None:
        printf('===== DESCRIBE INFO =====')
        printf(describe)

    printf('===== TRAIN INFO =====')
    printf('data dir = {}'.format(data_dir))
    printf('dropout = {}'.format(dropout))
    printf('epochs = {}'.format(epochs))
    printf('learn rate = {}'.format(learn_rate))
    printf('batch size = {}'.format(batch_size))
    printf('save dir = {}'.format(save_dir))
    printf('zip it = {}'.format(zipit))
    printf('on client = {}'.format(on_client))

    printf('===== DEVICE INFO =====')
    printf(str(device))
    printf('======================')

    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LEARN_RATE = learn_rate
    DROPOUT = dropout
    ON_CLIENT = on_client

    train_transorms, valid_transorms = make_transorms()
    train_dataiter, valid_dataset = get_datasets(
        data_dir, train_transorms, valid_transorms, batch_size)
    print('[INFO]generate datasets ok')

    if show_model is True:
        print(model)
        print()

    # RMSprop
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    best_valid_accuracy = -float('inf')
    for epoch in range(1, epochs+1):
        printf('Trainning: {}\n'.format(epoch))

        loss = train_nn(model, criterion, optimizer, train_dataiter, device=device)
        printf('Train Epoch: {}\t Loss: {:.6f}\n'.format(epoch, loss.item()))

        valid_loss, correct = valid(model, criterion, valid_dataset, device=device)
        printf('Valid set: Average of loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, len(valid_dataset), 100. * correct / len(valid_dataset)))

        curr_valid_accuracy = correct / len(valid_dataset)
        if (curr_valid_accuracy > best_valid_accuracy):
            printf('Now best model: Average of loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                valid_loss, correct, len(valid_dataset), 100. * correct / len(valid_dataset)))
            model_save_path = os.path.join(
                save_dir, 'model_state_dict.pth')
            torch.save(model.state_dict(), model_save_path)

    timestamp = time.asctime().replace(' ', '_').replace(':', '_')

    print_save_path = os.path.join(save_dir, timestamp+'.txt')
    f = open(print_save_path, 'w')
    for line in screen_print:
        f.write(line)
    f.close()

    new_model_save_path = os.path.join(
        save_dir, 'model_'+timestamp+'.pth')
    os.rename(model_save_path, new_model_save_path)
    print('save model to {}'.format(new_model_save_path))
    print('best valid accuracy: {:.4f}'.format(best_valid_accuracy))

    new_checkpoint_path = os.path.join(
        save_dir, 'model_'+timestamp+'.json')
    with open(new_checkpoint_path, "w") as f:
        state_dict = {
            'describe': describe,
            'best_valid_accuracy': best_valid_accuracy,
            'model_path': new_model_save_path,
            'info_path': print_save_path,
            'dropout': dropout
        }
        json.dump(state_dict, f)

    if zipit is True:
        tarfile_path = os.path.join(
            save_dir, 'model_'+timestamp+'.tar.gz')
        with tarfile.open(tarfile_path, "w:gz") as tar:
            tar.add(print_save_path)
            tar.add(new_checkpoint_path)
            tar.add(new_model_save_path)
            tar.close()

    return True

class TestDataset(Dataset):
    def __init__(self, data_dir, device=DEVICE):
        super(TestDataset, self).__init__()
        self.image_paths = glob(os.path.join(data_dir, '*.jpg'))
        _, test_transorms = make_transorms()
        self.transorms = test_transorms
        print('{} number of test image loaded.'.format(len(self.image_paths)))
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # assert index >= len(self.image_paths), 'index, out of range'

        path = self.image_paths[index]
        image = Image.open(path)
        image_tensor = self.transorms(image)
        image_tensor_gpu = torch.tensor(image_tensor, device=self.device)
        # image_tensor = transforms.functional.to_tensor(image)
        filename = os.path.basename(path)

        return (image_tensor_gpu, filename)


def generate_csv_handle(csv_path='./test.csv'):
    header = ['img', 'c0', 'c1', 'c2', 'c3',
              'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    csv_file = open(csv_path, 'w')
    csv_write = csv.writer(csv_file)
    csv_write.writerow(header)
    return csv_file, csv_write


def test(model, data_dir, csv_path='./result.csv', batch_size=BATCH_SIZE,
        on_client=ON_CLIENT, device=DEVICE):
    data_dir = os.path.abspath(data_dir)
    if checkpoint_path is not None:
        checkpoint_path = os.path.abspath(checkpoint_path)
    csv_path = os.path.abspath(csv_path)
    csv_file, csv_write = generate_csv_handle(csv_path)

    test_gen = TestDataset(os.path.join(data_dir, 'test/'), device=device)
    test_dataiter = DataLoader(test_gen, batch_size=batch_size, shuffle=False)
    print('[INFO]generate datasets ok')

    BATCH_SIZE = batch_size
    ON_CLIENT = on_client

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataiter):
            print('Test {}th batch'.format(i))
            print()

            images = data[0]
            filenames = data[1]

            pred = model(images)
            pred = torch.nn.functional.softmax(pred, dim=0)
            pred = pred.cpu().clone().detach().numpy()

            row = []
            for (filename, prob) in zip(filenames, pred):
                row.append(filename)
                row.extend(prob)
                csv_write.writerow(row)
                row.clear()

    csv_file.close()
    print('save csv to {}'.format(csv_path))

def run(command='train', data_dir=data_dir, checkpoint_path=None, epochs=EPOCHS,
        learn_rate=LEARN_RATE, dropout=DROPOUT, batch_size=BATCH_SIZE, save_dir='../model',
        show_model=False, describe=None, csv_path='./result.csv', zipit=False, on_client=ON_CLIENT):
    """
    command: train, test
    """
    DEVICE = device_setting(on_client)

    model_path = None
    if checkpoint_path is not None:
        with open(checkpoint_path, 'r') as load_f:
            config = json.load(load_f)
            model_path = config['model_path']

    model = make_model(model_path, dropout=dropout, on_client=on_client)
    model.to(device=DEVICE)

    if command == 'train':
        train(model, data_dir=data_dir, epochs=epochs, batch_size=batch_size,
              learn_rate=learn_rate, dropout=dropout, save_dir=save_dir,
              show_model=show_model, describe=describe, zipit=zipit, on_client=on_client,
              device=DEVICE)
        return True

    if command == 'test':
        test(model, data_dir=data_dir, batch_size=batch_size,
             csv_path=csv_path, on_client=on_client, device=DEVICE)
        return True

    print('invalid command')
    return False

# checkpoint_path=data_dir+'../model/masker_Sat_May__4_21_28_36_2019.json'
run(command='train', data_dir=data_dir, on_client=True)
# ========================================================================================================
if __name__ == '__main__':
    fire.Fire()
