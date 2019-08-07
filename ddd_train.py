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
import torchvision
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

import model_vgg16
import model_xception
import model_resnet152
import model_pnasnet5large

from ddd_units.image_preprocessing import calc_image_gradient_inside

ON_CLIENT = True
NUM_CLASSES = 10
EPOCHS = 10
LEARN_RATE = 0.001
DROPOUT = 0.5
DEVICE_ID = 0
CROP_SIZE = (320, 440)
#IMG_SIZE = (224, 224)
#IMG_SIZE = (229, 229)
IMG_SIZE = (331, 331)
BATCH_SIZE = 32
IMG_MEAN = (0.31633861, 0.38164441, 0.37510719)
IMG_STD = (0.28836174, 0.32873901, 0.33058995)
MAX_DEVICES = 4
DEFAULT_DEVICE_ID = 0
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cpu")

# ========================================================================================================

class DriverDataset(Dataset):
    def __init__(self, data_dir, transorms, drivers, 
                 dtype='train', fivecrop=False):
        super(DriverDataset, self).__init__()
        self.data_dir = data_dir
        self.transorms = transorms
        self.dtype = dtype
        self.image_paths = []
        self.labels = {}
        self.used_images = []

        #print('read "{}" drivers'.format(drivers))
        driver_imgs_list_csv = os.path.join(data_dir, "driver_imgs_list.csv")
        df = pd.read_csv(driver_imgs_list_csv)

        if drivers is None:
            self.drivers = set(df["subject"].data.obj)
        else:
            self.drivers = set(drivers)

        for driver in self.drivers:
            df_driver = df[(df["subject"] == driver)]
            #print('[INFO]loading "{}"...'.format(driver))
            for _, row in df_driver.iterrows():
                class_id = int(row['classname'][1])
                if class_id >= 0 or class_id <= 9:
                    filename = row["img"]
                    self.image_paths.append(filename)
                    self.labels[filename] = class_id
                    if fivecrop is True:
                        basename = os.path.basename(filename)
                        basename = basename[0:basename.rfind('.')]
                        for i in range(1, 6):
                            filename = basename + '_' + str(i) + '.jpg'
                            self.image_paths.append(filename)
                            self.labels[filename] = class_id
        print('[INFO]{} number of image loaded.'.format(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # assert index >= len(self.image_paths), 'index, out of range'
        self.used_images.append(index)
        filename = self.image_paths[index]
        label = self.labels[filename]
        image_path = os.path.join(self.data_dir, self.dtype,
                                  'c{}'.format(label), filename)
        image = Image.open(image_path)
        image_tensor = self.transorms(image)
        #show_image = torchvision.transforms.functional.to_pil_image(image_tensor)
        #show_image.show()
        return (image_tensor, label)

    def clear(self):
        self.used_images.clear()

def device_setting(on_client, id=0):
    if on_client is True:
        device_id = 'cuda:0'
    else:
        device_id = 'cuda:{}'.format(id)

    DEVICE = torch.device(device_id)
    #print('Device Id = {}'.format(DEVICE))
    return DEVICE


def make_model_setting(model_name='xception'):
    if model_name == 'vgg16':
        image_size = model_vgg16.IMG_SIZE
    elif model_name == 'xception':
        image_size = model_xception.IMG_SIZE
    elif model_name == 'pnasnet5':
        image_size = model_pnasnet5large.IMG_SIZE
    elif model_name == 'resnet152':
        image_size = model_resnet152.IMG_SIZE
    else:
        pass
    return image_size


def make_model(model_path=None, dropout=DROPOUT,
               num_classes=NUM_CLASSES, map_location=DEVICE,
               on_client=ON_CLIENT, model_name='xception'):
    model_names = ['vgg16', 'xception', 'pnasnet5', 'resnet152']
    if model_name not in model_names:
        print('[ERROR]model name is invaild')
        return None

    model = None
    print('[INFO]load "{}" model.'.format(model_name))
    if model_name == 'vgg16':
        IMG_SIZE = model_vgg16.IMG_SIZE
        model = model_vgg16.make_model(model_path=model_path, dropout=dropout,
                                       num_classes=num_classes, map_location=map_location, on_client=on_client)
    elif model_name == 'xception':
        IMG_SIZE = model_xception.IMG_SIZE
        model = model_xception.make_model(model_path=model_path, dropout=dropout,
                                          num_classes=num_classes, map_location=map_location, on_client=on_client)
    elif model_name == 'pnasnet5':
        IMG_SIZE = model_pnasnet5large.IMG_SIZE
        model = model_pnasnet5large.make_model(model_path=model_path, dropout=dropout,
                                               num_classes=num_classes, map_location=map_location, on_client=on_client)
    elif model_name == 'resnet152':
        IMG_SIZE = model_resnet152.IMG_SIZE
        model = model_resnet152.make_model(model_path=model_path, dropout=dropout,
                                           num_classes=num_classes, map_location=map_location, on_client=on_client)
    else:
        return None
    model = model.to(map_location)
    return model

def default_transorms(crop_size=CROP_SIZE, image_size=IMG_SIZE,
                      image_mean=IMG_MEAN, image_std=IMG_STD,
                      gray_scale=None, gradient_image=False):
    compose_functions = [transforms.CenterCrop(crop_size),
        transforms.Resize(image_size),
    ]

    if gray_scale is not None:
        gray_scale = int(gray_scale)
        compose_functions.append(transforms.Grayscale(
            num_output_channels=gray_scale))

    if gradient_image is True:
        compose_functions.append(transforms.Lambda(
            lambda image: calc_image_gradient_inside(image)))
    
    compose_functions.append(transforms.ToTensor())
    compose_functions.append(transforms.Normalize(image_mean, image_std))

    return transforms.Compose(compose_functions)

def make_transorms(random_sample=False, crop_size=CROP_SIZE,
                   image_size=IMG_SIZE, image_mean=IMG_MEAN, image_std=IMG_STD,
                   gray_scale=None, gradient_image=False, random_horizontal_flip_enabled=True,
                   random_resized_crop_enabled=True, random_affine_enabled=True, 
                   random_affine_degrees=10, random_affine_translate_low=0.1, 
                   random_affine_translate_high=0.3):

    train_transorms = None
    if random_sample is False:
        train_transorms = default_transorms(crop_size=crop_size, image_size=image_size,
                                            image_mean=image_mean, image_std=image_std,
                                            gray_scale=gray_scale, gradient_image=gradient_image)
    else:
        compose_functions = [transforms.CenterCrop(crop_size),
            transforms.Resize(image_size),
        ]

        if random_affine_enabled is True:
            compose_functions.append(transforms.RandomAffine(degrees=random_affine_degrees, 
                translate=(random_affine_translate_low, random_affine_translate_high)))

        if random_horizontal_flip_enabled is True:
            compose_functions.append(transforms.RandomHorizontalFlip())

        if random_resized_crop_enabled is True:
            compose_functions.append(transforms.RandomResizedCrop(image_size))

        if gray_scale is not None:
            gray_scale = int(gray_scale)
            compose_functions.append(transforms.Grayscale(
                num_output_channels=gray_scale))

        if gradient_image is True:
            compose_functions.append(transforms.Lambda(
                lambda image: calc_image_gradient_inside(image)))
        
        compose_functions.append(transforms.ToTensor())
        compose_functions.append(transforms.Normalize(image_mean, image_std))
        train_transorms = transforms.Compose(compose_functions)

    valid_transorms = default_transorms(crop_size=crop_size, image_size=image_size,
                                        image_mean=image_mean, image_std=image_std,
                                        gray_scale=gray_scale, gradient_image=gradient_image)
    return train_transorms, valid_transorms

"""
def get_datasets(train_data_dir, valid_data_dir=None,
                 train_transorms=None, valid_transorms=None,
                 batch_size=BATCH_SIZE, crop_size=CROP_SIZE,
                 image_size=IMG_SIZE, image_mean=IMG_MEAN,
                 image_std=IMG_STD):
    #train_data_dir = os.path.abspath(train_data_dir)
    #valid_data_dir = os.path.abspath(valid_data_dir)

    if train_transorms is None:
        train_transorms = default_transorms(crop_size=crop_size, image_size=image_size,
                                            image_mean=image_mean, image_std=image_std)

    train_gen = ImageFolder(train_data_dir, transform=train_transorms)
    if valid_transorms is not None:
        valid_gen = ImageFolder(valid_data_dir, transform=valid_transorms)
    train_dataloader = DataLoader(
        train_gen, batch_size=batch_size, shuffle=True)
    return train_dataloader, valid_gen
"""

def valid(model, criterion, valid_dataset, device=DEVICE):
    model.eval()

    loss = 0
    correct = 0

    valid_dataset.clear()
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

    for _, train_data in enumerate(train_dataiter):
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

def evaluate_model(curr_valid_accuracy, best_valid_accuracy,
                   curr_valid_loss, best_valid_loss, evaluate_type='accuracy'):
    if evaluate_type == 'accuracy':
        return True if curr_valid_accuracy > best_valid_accuracy else False

    if evaluate_type == 'loss':
        return True if curr_valid_loss < best_valid_loss else False

    return False


def _train(model, task_name, train_dataset, valid_dataset, basic_model_save_path=None, epochs=EPOCHS,
           batch_size=BATCH_SIZE, learn_rate=LEARN_RATE, dropout=DROPOUT, save_dir='../model',
           show_model=False, zipit=False, on_client=ON_CLIENT, last_accuracy=0.0, last_loss=float('inf'),
           device=DEVICE, evaluate_type='accuracy'):
    """
    trainning...
    """
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LEARN_RATE = learn_rate
    DROPOUT = dropout
    ON_CLIENT = on_client
    DEVICE = device

    if show_model is True:
        print(model)
        print()

    # RMSprop
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    model_save_path = None

    # log file create
    timestamp = time.asctime()
    start_timestamp = timestamp
    timestamp = timestamp.replace(' ', '_').replace(':', '_')
    log_path = os.path.join(save_dir, task_name+'_'+timestamp+'.log')
    log_f = open(log_path, 'w')

    def log_info(info):
        if info is None:
            print()
            log_f.write('\n')
        else:
            print(info)
            log_f.write(info)
            log_f.write('\n')

    log_info('[INFO]start at {}'.format(start_timestamp))

    # each file name is on start_timestamp
    start_timestamp = timestamp
    start_datetime = datetime.now()

    best_epoch = 0
    best_valid_loss = last_loss
    best_valid_accuracy = last_accuracy
    for epoch in range(1, epochs+1):
        log_info('Trainning: {}'.format(epoch))
        loss = train_nn(model, criterion, optimizer,
                        train_dataset, device=device)
        log_info('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))

        curr_valid_loss, correct = valid(
            model, criterion, valid_dataset, device=device)
        log_info('Valid set: Average of loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            curr_valid_loss, correct, len(valid_dataset), 100. * correct / len(valid_dataset)))
        curr_valid_accuracy = correct / len(valid_dataset)
        if evaluate_model(curr_valid_accuracy, best_valid_accuracy,
                          curr_valid_loss, best_valid_loss, evaluate_type) is True:
            log_info('Now best model: Average of loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                curr_valid_loss, correct, len(valid_dataset), 100. * correct / len(valid_dataset)))

            if model_save_path is not None:
                os.remove(model_save_path)

            # model save file name = 'task name' + '_' + timestamp
            timestamp = time.asctime().replace(' ', '_').replace(':', '_')
            model_save_path = os.path.join(
                save_dir, task_name+'_'+timestamp+'.pth')
            torch.save(model.state_dict(), model_save_path)
            best_valid_accuracy = curr_valid_accuracy
            best_valid_loss = curr_valid_loss
            best_epoch = epoch

    end_datetime = datetime.now()
    end_timestamp = time.asctime()
    spent_time = (end_datetime-start_datetime).seconds
    log_info('[INFO]end at {}'.format(end_timestamp))
    log_info('[INFO]train spent {} seconds'.format(spent_time))

    if model_save_path is None:
        # model_save_path is none and basic_model_save_path is None, it's not possible!!!
        # beacuse last_accuracy init value is 0.0.
        model_save_path = basic_model_save_path

    if basic_model_save_path is None:
        basic_model_save_path = model_save_path

    if best_valid_accuracy > last_accuracy:
        log_info('save model to {}'.format(model_save_path))
    elif basic_model_save_path is not None:
        log_info('use basic model save path {}'.format(basic_model_save_path))
    else:
        log_info('accuracy not improve')
    log_info('best valid accuracy: {:.4f}, best_valid_loss: {:.4f}, best_epoch: {:.4f}.'.format(
        best_valid_accuracy, best_valid_loss, best_epoch))

    new_checkpoint_path = os.path.join(
        save_dir, task_name+'_'+start_timestamp+'.json')
    with open(new_checkpoint_path, "w") as f:
        state_dict = {
            'task_name': task_name,
            'best_valid_accuracy': best_valid_accuracy,
            'best_valid_loss': best_valid_loss,
            'best_epoch': best_epoch,
            'basic_model_save_path': basic_model_save_path,
            'model_path': model_save_path,
            'log_path': log_path,
            'learn_rate': learn_rate,
            'batch_size': batch_size,
            'dropout': dropout,
            'spent_time': spent_time
        }
        json.dump(state_dict, f)

    log_f.close()

    if zipit is True:
        tarfile_path = os.path.join(
            save_dir, task_name+'_'+start_timestamp+'.tar.gz')
        with tarfile.open(tarfile_path, "w:gz") as tar:
            tar.add(log_path)
            tar.add(new_checkpoint_path)
            tar.add(model_save_path)
            tar.close()

    print('please use checkpoint file "{}"'.format(new_checkpoint_path))

def kfold_train(task_name, model_name, data_dir, kfold_path, checkpoint_path=None, epochs=EPOCHS,
                save_dir='../model', learn_rate=LEARN_RATE, dropout=DROPOUT, batch_size=BATCH_SIZE,
                show_model=False, zipit=False, on_client=ON_CLIENT, device=DEFAULT_DEVICE_ID, 
                max_devices=MAX_DEVICES, evaluate_type='accuracy', random_sample=False, fivecrop=False):

    if device < 0 or device > (max_devices-1):
        print('[ERROR]device = "{}" is invalid'.format(device))
        return
    DEVICE = device_setting(on_client, device)
    model_id = max_devices if device == 0 else device

    data_dir = os.path.abspath(data_dir)
    train_data_dir = os.path.join(data_dir, 'train')
    driver_imgs_list_csv = os.path.join(data_dir, "driver_imgs_list.csv")
    save_dir = os.path.abspath(save_dir)
    kfold_path = os.path.abspath(kfold_path)

    valid_drivers = None
    with open(kfold_path, 'r') as kfold_f:
        config = json.load(kfold_f)
        number_models = config['number_models']
        if model_id > number_models:
            print('[ERROR]model id = "{}" is over config number = {}'.format(
                model_id, number_models))
            return
        valid_drivers = config['valid_drivers']
        valid_drivers = config['valid_drivers']['model_{}'.format(model_id)]

    model_path = None
    last_accuracy = 0.0
    last_loss = float('inf')
    if checkpoint_path is not None:
        with open(checkpoint_path, 'r') as load_f:
            config = json.load(load_f)
            model_path = config['model_path']
            last_accuracy = config['best_valid_accuracy']
            last_loss = config['best_valid_loss']

    df = pd.read_csv(driver_imgs_list_csv)
    train_drivers = list(set(df["subject"]))
    for valid_driver in valid_drivers:
        train_drivers.remove(valid_driver)

    print('===== TRAIN INFO =====')
    print('task name = {}'.format(task_name))
    print('model name = {}'.format(model_name))
    print('model id = {}'.format(model_id))
    print('train data dir = {}'.format(train_data_dir))
    print('save dir = {}'.format(save_dir))
    print('kfold path = {}'.format(kfold_path))
    print('basic model save path = {}'.format(model_path))
    print('driver imgs list csv = {}'.format(driver_imgs_list_csv))
    print('dropout = {}'.format(dropout))
    print('epochs = {}'.format(epochs))
    print('learn rate = {}'.format(learn_rate))
    print('batch size = {}'.format(batch_size))
    print('zip it = {}'.format(zipit))
    print('on client = {}'.format(on_client))
    print('last accuracy = {}'.format(last_accuracy))
    print('last loss = {}'.format(last_loss))
    print('train drivers = {}'.format(train_drivers))
    print('valid drivers = {}'.format(valid_drivers))
    print('evaluate type = {}'.format(evaluate_type))
    print('random sample = {}'.format(random_sample))
    print('five crop = {}'.format(fivecrop))

    print('===== DEVICE INFO =====')
    print(str(device))

    image_size = make_model_setting(model_name=model_name)
    print('===== MODEL INFO =====')
    print('image size = {}'.format(image_size))
    print('======================')

    train_transorms, valid_transorms = make_transorms(
        random_sample=random_sample, image_size=image_size)
    train_dataset = DriverDataset(data_dir, train_transorms, train_drivers,
        fivecrop=fivecrop)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = DriverDataset(data_dir, valid_transorms, valid_drivers)
    print('[INFO]generate datasets ok')

    model = make_model(model_path, dropout=dropout, num_classes=NUM_CLASSES,
                       map_location=DEVICE, on_client=on_client, model_name=model_name)
    print('[INFO]generate model ok')

    _train(model, task_name=task_name, train_dataset=train_dataloader, valid_dataset=valid_dataset,
           basic_model_save_path=model_path, epochs=epochs, batch_size=batch_size, learn_rate=learn_rate,
           dropout=dropout, save_dir=save_dir, show_model=show_model, zipit=zipit, on_client=on_client,
           last_accuracy=last_accuracy, last_loss=last_loss, device=DEVICE, evaluate_type=evaluate_type)

# ========================================================================================================

class TestDataset(Dataset):
    def __init__(self, data_dir, test_transorms):
        super(TestDataset, self).__init__()
        self.image_paths = glob(os.path.join(data_dir, '*.jpg'))
        self.transorms = test_transorms
        print('{} number of test image loaded.'.format(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # assert index >= len(self.image_paths), 'index, out of range'
        path = self.image_paths[index]
        image = Image.open(path)
        image_tensor = self.transorms(image)
        #image_tensor_gpu = torch.tensor(image_tensor, device=self.device)
        # image_tensor = transforms.functional.to_tensor(image)
        filename = os.path.basename(path)
        return (image_tensor, filename)


def generate_csv_handle(csv_path='./test.csv'):
    header = ['img', 'c0', 'c1', 'c2', 'c3',
              'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    csv_file = open(csv_path, 'w')
    csv_write = csv.writer(csv_file)
    csv_write.writerow(header)
    return csv_file, csv_write


def bagging_valid(models, criterion, valid_dataset, devices):
    loss = 0
    correct = 0
    incorrects = []
    preds = []

    valid_dataset.clear()
    with torch.no_grad():
        for i, data in enumerate(valid_dataset):
            image = data[0]
            label = torch.tensor(data[1], dtype=torch.long)

            image = torch.unsqueeze(image, 0)
            label = torch.unsqueeze(label, 0)
            label_clone = label.clone()
            preds.clear()

            for model_id, model in enumerate(models):
                device = devices[model_id]
                image = image.to(device)
                label = label.to(device)
                pred = model(image)
                preds.append(pred.cuda().data.cpu().numpy())

            preds_tensor = torch.tensor(preds, dtype=torch.float64)
            preds_mean = preds_tensor.mean(dim=0)
            loss += criterion(preds_mean, label_clone).item()
            pred = preds_mean.max(1, keepdim=True)[1]
            num_corrects = pred.eq(label_clone.view_as(pred)).sum().item()
            if num_corrects == 0:
                incorrects.append((i, pred.data.numpy()[0][0]))
            correct += num_corrects
    loss /= len(valid_dataset)
    return loss, correct, preds_mean, incorrects

def bagging_test(data_dir, model_name, cpp1=None, cpp2=None,
                 cpp3=None, cpp4=None, batch_size=BATCH_SIZE, 
                 incorrects_log='./incorrects.log', valid_drivers=None):
    models = []
    devices = []

    image_size = make_model_setting(model_name=model_name)
    print('===== MODEL INFO =====')
    print('image size = {}'.format(image_size))
    print('======================')

    checkpoint_paths = [cpp1, cpp2, cpp3, cpp4]
    if any(checkpoint_paths) is False:
        print('[ERROR]all checkpoint files are empty')
        return

    checkpoint_paths = [cpp for cpp in checkpoint_paths if cpp is not None]
    for i, checkpoint_path in enumerate(checkpoint_paths):
        model_path = None
        print('[INFO]read checkpoint: {}'.format(checkpoint_path))
        with open(checkpoint_path, 'r') as load_f:
            config = json.load(load_f)
            model_path = config['model_path']
            print('      read model: {}'.format(model_path))
        device_id = 'cuda:{}'.format(i)
        device = torch.device(device_id)
        model = make_model(model_path, on_client=False, model_name=model_name,
                           map_location=device)
        model = model.to(device)
        model.eval()
        models.append(model)
        devices.append(device)

    print('[INFO]models generated')

    valid_transorms = default_transorms(image_size=image_size)
    #valid_dataset = ImageFolder(valid_data_dir, transform=valid_transorms)
    valid_dataset = DriverDataset(data_dir, valid_transorms, valid_drivers)
    print('[INFO]generate datasets ok')

    # output incorrects to file
    inc_f = open(incorrects_log, 'w')

    def incorrect_record(info=None, records=None):
        if info is not None:
            inc_f.write(info)
            inc_f.write('\n')

        if records is not None:
            for record in records:
                read_index = record[0]
                pred = record[1]
                data_index = valid_dataset.used_images[read_index]
                filename = valid_dataset.image_paths[data_index]
                label = valid_dataset.labels[filename]
                text = '{}:{}:{}\n'.format(filename, label, pred)
                inc_f.write(text)

    criterion = torch.nn.CrossEntropyLoss()
    loss, correct, _, incorrects = bagging_valid(
        models, criterion, valid_dataset, devices)
    print('Valid set: Average of loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(valid_dataset), 100. * correct / len(valid_dataset)))
    incorrect_record(records=incorrects)
    inc_f.close()

def kaggle_bagging_test(test_data_dir, model_name, cpp1=None, cpp2=None,
                        cpp3=None, cpp4=None, csv_path='./result.csv',
                        batch_size=BATCH_SIZE):
    models = []
    devices = []

    image_size = make_model_setting(model_name=model_name)
    print('===== MODEL INFO =====')
    print('image size = {}'.format(image_size))
    print('======================')

    checkpoint_paths = [cpp1, cpp2, cpp3, cpp4]
    if any(checkpoint_paths) is False:
        print('[ERROR]all checkpoint files are empty')
        return

    checkpoint_paths = [cpp for cpp in checkpoint_paths if cpp is not None]
    for i, checkpoint_path in enumerate(checkpoint_paths):
        model_path = None
        print('[INFO]read checkpoint: {}'.format(checkpoint_path))
        with open(checkpoint_path, 'r') as load_f:
            config = json.load(load_f)
            model_path = config['model_path']
            print('      read model: {}'.format(model_path))
        # select device id
        device_id = 'cuda:{}'.format(i)
        device = torch.device(device_id)
        model = make_model(model_path, on_client=False, model_name=model_name,
                           map_location=device)
        model.eval()
        models.append(model)
        devices.append(device)
    print('[INFO]models generated')

    csv_path = os.path.abspath(csv_path)
    csv_file, csv_write = generate_csv_handle(csv_path)
    print('[INFO]create csvfile ok')

    test_data_dir = os.path.abspath(test_data_dir)
    test_transorms = default_transorms(image_size=image_size)
    test_gen = TestDataset(test_data_dir, test_transorms=test_transorms)
    test_dataiter = DataLoader(test_gen, batch_size=batch_size, shuffle=False)
    print('[INFO]generate datasets ok')

    preds = []
    row = []
    with torch.no_grad():
        for i, data in enumerate(test_dataiter):
            if i % 1000 == 0:
                print('running on {}th batch'.format(i))
            image = data[0]
            filenames = data[1]

            preds.clear()

            for j, model in enumerate(models):
                device = devices[j]
                image = image.to(device)
                pred = model(image)
                pred = torch.nn.functional.softmax(pred, dim=0)
                #pred = pred.cuda().data.cpu().numpy()
                preds.append(pred)

            preds_tensor = torch.stack(preds, dim=0)
            preds_mean = preds_tensor.mean(dim=0)
            preds_mean = preds_mean.cuda().data.cpu().numpy()
            #pred = preds_mean.max(1, keepdim=True)[1]

            row.clear()
            for (filename, probs) in zip(filenames, preds_mean):
                row.append(filename)
                row.extend(probs)
                csv_write.writerow(row)
                row.clear()

    csv_file.close()
    print('save csv to {}'.format(csv_path))


#kfold_train(task_name='test', model_name='xception', data_dir='../data', kfold_path='../data/kfold_bagging.json')
if __name__ == '__main__':
    fire.Fire()
