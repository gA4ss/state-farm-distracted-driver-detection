# coding:utf-8
"""
"""
import os
import sys
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
MASKER_KEEP_LAYER = None
FAKE_TRAIN = False
GENERATE_MASK_IMAGE = False

CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if CUDA else "cpu")

DEVICE_MASKER = None
DEVICE_CLASSIFIER = None
DEVICE_DEBUG = None
DEVICE_TEST = None

def device_setting(on_client):
    if on_client is True:
        DEVICE_MASKER = torch.device("cuda:0")
        DEVICE_CLASSIFIER = torch.device("cpu")
        DEVICE_DEBUG = torch.device("cuda:0")
        DEVICE_TEST = torch.device("cpu")
    else:
        DEVICE_MASKER = torch.device("cuda:1")
        DEVICE_CLASSIFIER = torch.device("cuda:2")
        DEVICE_DEBUG = torch.device("cuda:3")
        DEVICE_TEST = torch.device("cpu")

ddd_mask_images = [torch.zeros(IMG_SIZE)] * 10

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
    def __init__(self, model=None, dropout=DROPOUT, num_classes=NUM_CLASSES, masker_keep_layer=None):
        super(MaskNet, self).__init__()

        if model is None:
            self.model = VGGNet()
        else:
            self.model = model
        # self.model.to(DEVICE_MASKER)

        self.dropout = dropout
        self.num_classes = num_classes
        self.masker_keep_layer = masker_keep_layer

        for name, param in self.model.named_parameters():
            param.requires_grad = self.keep_layer(name)

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

        #self.masker = nn.Conv2d(32, 3, kernel_size=1)
        self.masker = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Conv2d(16, 8, kernel_size=1),
            nn.Conv2d(8, 3, kernel_size=1)
        )

        self.excessive = nn.Conv2d(3, 1, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(50176, 4096),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, self.num_classes)
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
        self.mask = self.masker(mask)

        # use filter to mask
        #self.tensor_filter(self.mask)
        
        #mask_cpu = self.mask.cpu()
        #mask = transforms.functional.to_pil_image(mask_cpu)
        # img.show()

        self.feature_image = self.excessive(self.mask)
        x = self.feature_image.view(self.feature_image.size(0), -1)
        pred = self.classifier(x)

        return pred

    def tensor_filter(self, tensors_items):
        for item in tensors_items:
            for tensor in item:
                mean = tensor.mean()
                tensor[tensor<mean] = 0

    def keep_layer(self, name):
        if self.masker_keep_layer is None:
            return False

        if len(self.masker_keep_layer) == 0:
            return False

        for kl in self.masker_keep_layer:
            if kl in name:
                return True
        return False


class ClassifierNet(nn.Module):
    def __init__(self, model=None, dropout=DROPOUT, num_classes=NUM_CLASSES):
        super(ClassifierNet, self).__init__()

        if model is None:
            self.model = pretrainedmodels.__dict__[
                'xception'](pretrained='imagenet')
        else:
            self.model = model
        # self.model.to(DEVICE_CLASSIFIER)

        self.dropout = dropout
        self.num_classes = num_classes

        self.classifier_transorms = transforms.Compose([
            transforms.Lambda(lambda tensors:
                              [transforms.Resize((299, 299))(transforms.ToPILImage()(tensor)) for tensor in tensors]),
            transforms.Lambda(lambda tensors:
                              torch.stack([transforms.ToTensor()(tensor) for tensor in tensors]))
        ])

        # extract freature
        # self.feature = nn.Sequential(*list(self.model.children())[:-1])

        for param in self.model.parameters():
            param.requires_grad = False

        in_dim = self.model.last_linear.in_features

        """
        finetune = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(in_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(1000, self.num_classes)
        )
        """
        finetune = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(in_dim, 1000),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(1000, self.num_classes)
        )

        self.model.last_linear = finetune

    def forward(self, x):
        x = x.cpu()
        x = self.classifier_transorms(x)
        x = torch.as_tensor(x, device=DEVICE_CLASSIFIER)
        x = self.model(x)
        return x


# ========================================================================================================


def make_transorms():
    train_transorms = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        #transforms.RandomAffine(degrees=10, translate=(0.1, 0.3)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomResizedCrop(IMG_SIZE),
        transforms.Resize(IMG_SIZE),
        # transforms.FiveCrop(CROP_SIZE),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
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


def get_datasets(data_dir, train_transorms, valid_transorms=None, batch_size=BATCH_SIZE):
    valid_gen = None
    train_gen = ImageFolder(os.path.join(
        data_dir, 'train/'), transform=train_transorms)
    if valid_transorms is not None:
        valid_gen = ImageFolder(os.path.join(
            data_dir, 'valid/'), transform=valid_transorms)

    train_dataiter = DataLoader(train_gen, batch_size=batch_size, shuffle=True)
    return train_dataiter, valid_gen


def make_model(masker_path=None, classifier_path=None,
               masker_dropout=DROPOUT, classifier_dropout=DROPOUT,
               masker_keep_layer=MASKER_KEEP_LAYER, on_client=ON_CLIENT):
    masker_model = MaskNet(
        dropout=masker_dropout, num_classes=NUM_CLASSES, masker_keep_layer=masker_keep_layer)
    classifier_model = ClassifierNet(
        dropout=classifier_dropout, num_classes=NUM_CLASSES)

    if masker_path is not None:
        if on_client:
            masker_model.load_state_dict(torch.load(
                masker_path, map_location=torch.device('cpu')))
        else:
            masker_model.load_state_dict(torch.load(masker_path))

    if classifier_path is not None:
        if on_client:
            classifier_model.load_state_dict(torch.load(
                classifier_path, map_location=torch.device('cpu')))
        else:
            classifier_model.load_state_dict(torch.load(classifier_path))

    return masker_model, classifier_model


def valid(masker_model, classifier_model, masker_criterion, classifier_criterion, datasets):
    masker_model.eval()
    classifier_model.eval()

    loss = 0
    correct = 0

    mask_loss = 0
    mask_correct = 0

    with torch.no_grad():
        for i, data in enumerate(datasets):
            image = data[0]
            label = torch.tensor(data[1], dtype=torch.long)

            image = torch.unsqueeze(image.to(DEVICE_MASKER), 0)
            label = torch.unsqueeze(label.to(DEVICE_CLASSIFIER), 0)
            mask_label = torch.as_tensor(label, device=DEVICE_MASKER)

            masker_pred = masker_model(image)
            mask_loss += masker_criterion(masker_pred, mask_label).item()
            masker_pred = masker_pred.max(1, keepdim=True)[1]
            mask_correct += masker_pred.eq(
                mask_label.view_as(masker_pred)).sum().item()

            masker_output = masker_model.mask
            classifier_input = torch.as_tensor(
                masker_output, device=DEVICE_CLASSIFIER)
            classifier_pred = classifier_model(classifier_input)

            loss += classifier_criterion(classifier_pred, label).item()
            pred = classifier_pred.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    loss /= len(datasets)
    mask_loss /= len(datasets)
    return loss, correct, mask_loss, mask_correct


def train_nn(masker_model, classifier_model, masker_criterion, classifier_criterion,
             masker_optimizer, classifier_optimizer, train_dataiter, fake_masker=False,
             generate_mask_image=GENERATE_MASK_IMAGE):
    masker_model.train()
    classifier_model.train()

    for batch_index, train_data in enumerate(train_dataiter):
        train_images_masker = train_data[0]
        train_labels_masker = train_data[1]
        train_labels_classifier = train_data[1]

        # copy CPU data to GPU
        train_images_masker = train_images_masker.to(DEVICE_MASKER)
        train_labels_masker = train_labels_masker.to(DEVICE_MASKER)

        train_labels_classifier = train_labels_classifier.to(DEVICE_CLASSIFIER)

        masker_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        masker_pred = masker_model(train_images_masker)
        masker_output = masker_model.mask

        if generate_mask_image:
            masker_feature_image = masker_model.feature_image
            for i, feature in enumerate(masker_feature_image):
                label = train_data[1][i]
                feature_cpu = feature.cpu()
                assert ddd_mask_images[label].size() == feature_cpu.size()
                ddd_mask_images[label] += feature_cpu

        classifier_input = torch.as_tensor(
            masker_output, device=DEVICE_CLASSIFIER)
        classifier_pred = classifier_model(classifier_input)

        classifier_loss = classifier_criterion(
            classifier_pred, train_labels_classifier)

        # use classifier's pred to train masknet
        if fake_masker is True:
            fake_masker_pred = torch.as_tensor(
                classifier_pred, device=DEVICE_MASKER)
            masker_loss = masker_criterion(
                fake_masker_pred, train_labels_masker)
        else:
            masker_loss = masker_criterion(masker_pred, train_labels_masker)

        classifier_loss.backward(retain_graph=True)
        masker_loss.backward()

        classifier_optimizer.step()
        masker_optimizer.step()
    return masker_loss, classifier_loss


def train(data_dir=data_dir, epochs=EPOCHS, batch_size=BATCH_SIZE,
          masker_path=None, classifier_path=None, masker_learn_rate=LEARN_RATE, classifier_learn_rate=LEARN_RATE,
          masker_dropout=DROPOUT, classifier_dropout=DROPOUT, masker_keep_layer=MASKER_KEEP_LAYER, fake_train=FAKE_TRAIN,
          generate_mask_image=GENERATE_MASK_IMAGE, save_dir='../model', show_model=False, describe=None,
          zipit=False, on_client=ON_CLIENT):
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
    if masker_path is not None:
        masker_path = os.path.abspath(masker_path)
    if classifier_path is not None:
        classifier_path = os.path.abspath(classifier_path)
    save_dir = os.path.abspath(save_dir)
    fake_train = bool(fake_train)

    if masker_keep_layer is not None:
        if isinstance(masker_keep_layer, int):
            masker_keep_layer = [masker_keep_layer]
        elif isinstance(masker_keep_layer, str):
            masker_keep_layer = [int(x) for x in masker_keep_layer.split(',')]
        elif isinstance(masker_keep_layer, list):
            pass
        else:
            masker_keep_layer = None

    if describe is not None:
        printf('===== DESCRIBE INFO =====')
        printf(describe)

    printf('===== TRAIN INFO =====\n')
    printf('data dir = {}\n'.format(data_dir))
    printf('masker path = {}\n'.format(
        masker_path if masker_path is not None else '[new model]'))
    printf('classifier path = {}\n'.format(
        classifier_path if classifier_path is not None else '[new model]'))
    printf('masker dropout = {}\n'.format(masker_dropout))
    printf('classifier dropout = {}\n'.format(classifier_dropout))
    printf('masker keep layers = {}\n'.format(masker_keep_layer))
    printf('epochs = {}\n'.format(epochs))
    printf('masker learn rate = {}\n'.format(masker_learn_rate))
    printf('classifier learn rate = {}\n'.format(classifier_learn_rate))
    printf('batch size = {}\n'.format(batch_size))
    printf('fake_train = {}\n'.format(fake_train))
    printf('save dir = {}\n'.format(save_dir))
    printf('======================\n')

    train_transorms, valid_transorms = make_transorms()
    train_dataiter, valid_datasets = get_datasets(
        data_dir, train_transorms, valid_transorms, batch_size)
    print('[INFO]generate datasets ok')

    masker_model, classifier_model = make_model(
        masker_path, classifier_path, masker_dropout, classifier_dropout,
        masker_keep_layer, on_client)
    masker_model.to(DEVICE_MASKER)
    classifier_model.to(DEVICE_CLASSIFIER)
    print('[INFO]load model ok')

    # for name, param in classifier_model.named_parameters():
    #    if param.requires_grad is True:
    #        print('trainning --- {}'.format(name))
    #    else:
    #        print(name)
    #    print(param.grad)

    if show_model is True:
        print(masker_model)
        print()
        print(classifier_model)

    # RMSprop
    masker_criterion = torch.nn.CrossEntropyLoss()
    masker_optimizer = torch.optim.Adam(
        masker_model.parameters(), lr=masker_learn_rate)

    classifier_criterion = torch.nn.CrossEntropyLoss()
    classifier_optimizer = torch.optim.Adam(classifier_model.parameters(),
                                            lr=classifier_learn_rate)

    best_valid_accuracy = -float('inf')
    for epoch in range(1, epochs+1):
        printf('Trainning: {}\n'.format(epoch))

        masker_train_loss, classifier_train_loss = train_nn(
            masker_model, classifier_model, masker_criterion, classifier_criterion,
            masker_optimizer, classifier_optimizer, train_dataiter, fake_train,
            generate_mask_image)

        printf('Train Epoch: {}\t Masker Loss: {:.6f}\t Classifier Loss: {:.6f}\n'.format(
            epoch, masker_train_loss.item(), classifier_train_loss.item()))

        valid_loss, correct, mask_loss, mask_correct = valid(
            masker_model, classifier_model, masker_criterion, classifier_criterion, valid_datasets)

        printf('Valid set: Average of masker loss: {:.4f}, Accuracy of masker: {}/{} ({:.0f}%)\n'.format(
            mask_loss, mask_correct, len(valid_datasets), 100. * mask_correct / len(valid_datasets)))

        printf('Valid set: Average of classifier loss: {:.4f}, Accuracy of classifier: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, len(valid_datasets), 100. * correct / len(valid_datasets)))

        curr_valid_accuracy = correct / len(valid_datasets)
        if (curr_valid_accuracy > best_valid_accuracy):
            masker_save_path = os.path.join(
                save_dir, 'xvgg_masker_state_dict.pth')
            torch.save(masker_model.state_dict(), masker_save_path)

            classifier_save_path = os.path.join(
                save_dir, 'xvgg_classifier_state_dict.pth')
            torch.save(classifier_model.state_dict(), classifier_save_path)

    timestamp = time.asctime().replace(' ', '_').replace(':', '_')

    print_save_path = os.path.join(save_dir, timestamp+'.txt')
    f = open(print_save_path, 'w')
    for line in screen_print:
        f.write(line)
    f.close()

    new_masker_save_path = os.path.join(
        save_dir, 'xvgg_masker_'+timestamp+'.pth')
    os.rename(masker_save_path, new_masker_save_path)
    print('save masker model to {}'.format(new_masker_save_path))

    new_classifier_save_path = os.path.join(
        save_dir, 'xvgg_classifier_'+timestamp+'.pth')
    os.rename(classifier_save_path, new_classifier_save_path)
    print('save classifier model to {}'.format(new_classifier_save_path))

    new_checkpoint_path = os.path.join(save_dir, 'xvgg_'+timestamp+'.json')
    with open(new_checkpoint_path, "w") as f:
        state_dict = {
            'describe': describe,
            'masker_path': new_masker_save_path,
            'classifier_path': new_classifier_save_path,
            'info_path': print_save_path,
            'masker_dropout': masker_dropout,
            'classifier_dropout': classifier_dropout,
            'masker_keep_layer': masker_keep_layer,
            'fake_train': fake_train
        }
        json.dump(state_dict, f)

    mask_image_path = None
    if generate_mask_image is True:
        img_dict = {}
        mask_image_path = os.path.join(save_dir, 'xvgg_mask_image'+timestamp+'.npy')
        for i, img in enumerate(ddd_mask_images):
            img_dict['c{}'.format(i)] = img.numpy()
        np.save(mask_image_path, img_dict)

    if zipit is True:
        tarfile_path = os.path.join(save_dir, 'xvgg_'+timestamp+'.tar.gz')
        with tarfile.open(tarfile_path, "w:gz") as tar:
            tar.add(print_save_path)
            tar.add(new_checkpoint_path)
            tar.add(new_masker_save_path)
            tar.add(new_classifier_save_path)
            if mask_image_path is not None:
                tar.add(mask_image_path)
            tar.close()

    return True

def model_print():
    model = MaskNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    with SummaryWriter(log_dir='../ddd_model', comment='Masknet')as w:
        w.add_graph(model, dummy_input)

# ========================================================================================================


class TestDataset(Dataset):
    def __init__(self, data_dir):
        super(TestDataset, self).__init__()
        self.image_paths = glob(os.path.join(data_dir, '*.jpg'))
        _, test_transorms = make_transorms()
        self.transorms = test_transorms
        print('{} number of test image loaded.'.format(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # assert index >= len(self.image_paths), 'index, out of range'

        path = self.image_paths[index]
        image = Image.open(path)
        image_tensor = self.transorms(image)
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


def test(data_dir=data_dir, masker_path=None, classifier_path=None, batch_size=256,
         csv_path='./result.csv', on_client=ON_CLIENT):
    data_dir = os.path.abspath(data_dir)
    if masker_path is not None:
        masker_path = os.path.abspath(masker_path)
    if classifier_path is not None:
        classifier_path = os.path.abspath(classifier_path)
    csv_path = os.path.abspath(csv_path)

    csv_file, csv_write = generate_csv_handle(csv_path)

    print('===== TEST INFO =====')
    print('data dir = {}'.format(data_dir))
    print('masker path = {}'.format(masker_path))
    print('classifier path = {}'.format(classifier_path))
    print('batch size = {}'.format(batch_size))
    print('csv path = {}'.format(csv_path))
    print('======================')

    print()

    test_gen = TestDataset(os.path.join(data_dir, 'test/'))

    if on_client is True:
        batch_size = 3
    test_dataiter = DataLoader(test_gen, batch_size=batch_size, shuffle=False)
    print('[INFO]generate datasets ok')

    masker_model, classifier_model = make_model(masker_path, classifier_path)

    # named_parameters, named_children, named_modules
    # for name, child in masker_model.named_parameters():
    #    print(name)

    masker_model.to(DEVICE_TEST)
    classifier_model.to(DEVICE_TEST)
    print('[INFO]load model ok')

    # extract_image(masker_model, 0)

    masker_model.eval()
    classifier_model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_dataiter):
            print('Test {}th batch'.format(i))
            print()

            images = data[0]
            filenames = data[1]

            masker_pred = masker_model(images)
            masker_output = masker_model.mask

            # masker_images = masker_model.excessive(masker_output)
            # masker_image = transforms.functional.to_pil_image(masker_output[0])
            # masker_image.show()

            classifier_pred = classifier_model(masker_output)

            prob_list = torch.nn.functional.softmax(classifier_pred, dim=0)
            prob_list = prob_list.numpy()
            row = []
            for (filename, prob) in zip(filenames, prob_list):
                row.append(filename)
                row.extend(prob)
                csv_write.writerow(row)
                row.clear()

    csv_file.close()
    print('save csv to {}'.format(csv_path))

# ========================================================================================================


def run(command='train', data_dir=data_dir, checkpoint_path=None, epochs=EPOCHS,
        masker_learn_rate=LEARN_RATE, classifier_learn_rate=LEARN_RATE,
        masker_dropout=DROPOUT, classifier_dropout=DROPOUT, masker_keep_layer=MASKER_KEEP_LAYER,
        fake_train=FAKE_TRAIN, generate_mask_image=GENERATE_MASK_IMAGE, batch_size=BATCH_SIZE, save_dir='../model',
        show_model=False, describe=None, csv_path='./result.csv', zipit=False, on_client=ON_CLIENT):
    """
    command: train, test
    """
    device_setting(on_client)

    masker_path = None
    classifier_path = None
    if checkpoint_path is not None:
        with open(checkpoint_path, 'r') as load_f:
            config = json.load(load_f)
            masker_path = config['masker_path']
            classifier_path = config['classifier_path']
            masker_keep_layer = config['masker_keep_layer']

    if command == 'train':
        train(data_dir=data_dir, epochs=epochs, batch_size=batch_size,
              masker_path=masker_path, classifier_path=classifier_path,
              masker_learn_rate=masker_learn_rate, classifier_learn_rate=classifier_learn_rate,
              masker_dropout=masker_dropout, classifier_dropout=classifier_dropout,
              masker_keep_layer=masker_keep_layer, fake_train=fake_train,
              save_dir=save_dir, show_model=show_model, describe=describe, zipit=zipit, on_client=on_client)
        return True

    if command == 'test':
        test(data_dir=data_dir, masker_path=masker_path, classifier_path=classifier_path,
             batch_size=batch_size, csv_path=csv_path, on_client=on_client)
        return True

    print('invalid command')
    return False

# ========================================================================================================


#test(data_dir=data_dir+'../data', csv_path='./result.csv')
if __name__ == '__main__':
    fire.Fire()
