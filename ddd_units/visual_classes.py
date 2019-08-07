# coding:utf-8
"""
显示从c0 - c9 一行一个状态，每个状态6张图片
图片从训练集中随机获取
"""
import os
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

data_dir = "../data/"
subdir = "train"

driver_imgs_list_csv = os.path.join(data_dir, "driver_imgs_list.csv")
df = pd.read_csv(driver_imgs_list_csv)

def driving_status_image():
    img_list = {}
    for i in range(10):
        train_dir = os.path.join(data_dir, "train", "c%d"%i)
        image_files = glob.glob(os.path.join(train_dir,"*.jpg"))
        files_count = len(image_files)
        select_image = random.sample(range(files_count), 6)

        status = 'c{}'.format(i)
        img_list[status] = []
        for select in select_image:
            img_path = image_files[select]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list[status].append(img)

    plt.figure(figsize=(6, 10))
    plt.suptitle('driving status')

    for status_id in range(10):
        status = 'c{}'.format(status_id)
        status_imgs = img_list[status]
        for i, img in enumerate(status_imgs):
            if i == 0:
                index = status_id+1
            else:
                index = (status_id+1)+10*i
            plt.subplot(6, 10, index)
            if i == 0:
                plt.title(status)
            plt.imshow(img)
            plt.axis('off')
    plt.show()

driving_status_image()