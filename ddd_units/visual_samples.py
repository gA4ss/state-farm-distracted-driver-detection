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

def driving_status_image(show_colums=6):
    sample_list = []
    for i in range(10):
        train_dir = os.path.join(data_dir, "train", "c%d"%i)
        image_files = glob.glob(os.path.join(train_dir,"*.jpg"))
        files_count = len(image_files)
        # 随机选择6张
        select_images = random.sample(range(files_count), show_colums)
        read_images = []
        # 循环读取图像
        for j in select_images:
            img = cv2.imread(image_files[j], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_info = (img, image_files[j])
            read_images.append(image_info)
        # 添加到列表
        sample_list.append(read_images)

    plt.figure(figsize=(10, show_colums))
    plt.suptitle('driving status')

    for i in range(10 * show_colums):
        # 显示驾驶员编号
        img_info = sample_list[(i-1) // show_colums][(i-1) % show_colums]
        img_name = os.path.basename(img_info[1])
        title = df[df["img"] == img_name]["subject"].values[0]
        plt.subplot(10, show_colums, i+1)
        plt.title(title)
        plt.axis('off')
        plt.imshow(img_info[0])
    plt.show()

driving_status_image()