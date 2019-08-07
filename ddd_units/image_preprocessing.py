# coding:utf-8
import os
import cv2
import numpy as np
import fire
from glob import glob
from PIL import Image
from torchvision import transforms

CROP_SIZE = (320, 440)

def generate_five_crop(image_path):
    image_path = os.path.abspath(image_path)
    dirname = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    filename = filename[0:filename.rfind('.')]

    five_crop = transforms.FiveCrop(CROP_SIZE)
    image = Image.open(image_path)
    images = five_crop(image)
    for i, img in enumerate(images):
        img_path = os.path.join(dirname, filename)
        img_path += '_' + str(i+1) + '.jpg'
        img.save(img_path)

def generate_five_crop_dir(data_dir):
    data_dir = os.path.abspath(data_dir)
    for i in range(10):
        train_dir = os.path.join(data_dir, "train", "c%d"%i)
        image_paths = glob(os.path.join(train_dir, "*.jpg"))
        for image_path in image_paths:
            print('handling "{}" file'.format(image_path))
            generate_five_crop(image_path)

def calc_image_gradient_inside(image, X_weight=0.5, Y_weight=0.5):
    image = np.array(image)

    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, X_weight, absY, Y_weight, 0)
    return dst

def calc_image_gradient(image_path, show=True):
    img = Image.open(image_path)
    img = np.array(img)
    #img = cv2.imread(image_path, 0)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    
    if show is True:
        while(1):
            cv2.imshow("absX", absX)
            cv2.imshow("absY", absY)
            cv2.imshow("Result", dst)
            k = cv2.waitKey(1) & 0XFF
            if k==ord('q'):
                break;
        cv2.destroyAllWindows()

    return dst

if __name__ == '__main__':
    #generate_five_crop_dir('../data')
    fire.Fire()