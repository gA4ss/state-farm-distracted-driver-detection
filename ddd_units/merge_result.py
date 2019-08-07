# coding:utf-8
import os
import sys
import csv
import fire
import time
import json

from glob import glob
from PIL import Image
from datetime import datetime
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_csv_handle(csv_path='./merge_result.csv'):
    header = ['img', 'c0', 'c1', 'c2', 'c3',
              'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    csv_file = open(csv_path, 'w')
    csv_write = csv.writer(csv_file)
    csv_write.writerow(header)
    return csv_file, csv_write

def merge_result_files(file1_path, file2_path, 
                       merge_result='../data/merge_result.csv'):
    if merge_result is not None:
        merge_result = os.path.abspath(merge_result)
    csv_file, csv_write = generate_csv_handle(merge_result)

    file1_path = os.path.abspath(file1_path)
    file2_path = os.path.abspath(file2_path)

    result1_df = pd.read_csv(file1_path)
    result2_df = pd.read_csv(file2_path)

    images = []
    socres = []
    
    for row1, row2 in zip(result1_df.iterrows(), result2_df.iterrows()):
        if row1[1]['img'] != row2[1]['img']:
            print('{} - {} is not same'.format(row1[1]['img'], row2[1]['img']))
            return

        image = row1[1]['img']
        print('handle "{}"'.format(image))
        images.append(image)

        each_socres = []
        for i in range(10):
            socre1 = row1[1]['c'+str(i)]
            socre2 = row2[1]['c'+str(i)]
            socre = 1 / 2 * (socre1 + socre2)
            each_socres.append(socre)
        print(each_socres)
        socres.append(each_socres)

    row = []
    for (image, socre) in zip(images, socres):
        row.append(image)
        row.extend(socre)
        csv_write.writerow(row)
        row.clear()

    csv_file.close()
    print('save csv to {}'.format(merge_result))



merge_result_files('./result.csv', './result_5.csv', '../data/result_6.csv')
if __name__ == '__main__':
    fire.Fire()