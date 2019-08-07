# coding:utf-8

import os
import pandas as pd
import numpy as np
import shutil
import random
import inspect
from glob import glob

def splite_valid():

    base_dir = os.path.abspath(os.path.join(inspect.getfile(splite_valid), os.pardir))
    data_dir = os.path.join(base_dir, "../../data/")

    driver_imgs_list_csv = os.path.join(data_dir, "driver_imgs_list.csv")
    df = pd.read_csv(driver_imgs_list_csv)
    #subjects = list(set(df["subject"]))
    #select = random.sample(range(len(subjects)), 2)
    valid_subjects = ['p002', 'p047']

    print(valid_subjects)

    if not os.path.exists(data_dir + "valid"):
        os.mkdir(data_dir + "valid")
        for i in range(10):
            os.mkdir(data_dir + "valid/c%d"%i)

    for valid_subject in valid_subjects:
        df_valid = df[(df["subject"]==valid_subject)]
        for index, row in df_valid.iterrows():
            subpath = row["classname"] + "/" + row["img"]
            if os.path.exists(os.path.join(data_dir,"train",subpath)):
                shutil.move(os.path.join(data_dir,"train",subpath), os.path.join(data_dir,"valid",subpath),)
            else:
                print("cannot move {} : {}".format(row["subject"],subpath))

def data_recover():
    base_dir = os.path.abspath(os.path.join(inspect.getfile(data_recover), os.pardir))
    data_dir = os.path.join(base_dir, "../../data/")
    for i in range(10):
        path = data_dir + "valid/c%d/"%i
        file_paths = glob(os.path.join(path, "*.jpg"))
        for path in file_paths:
            file_name = os.path.basename(path)
            to_file_path = data_dir + "train/c%d/%s" % (i, file_name)
            #print("move %s to %s"%(path, to_file_path))
            shutil.move(path, to_file_path,)


def bagging_splite(k=4):

    base_dir = os.path.abspath(os.path.join(inspect.getfile(splite_valid), os.pardir))
    data_dir = os.path.join(base_dir, "../../data/")

    driver_imgs_list_csv = os.path.join(data_dir, "driver_imgs_list.csv")
    df = pd.read_csv(driver_imgs_list_csv)
    driver_list = set(df['subject'])
    print('Drivers {}:\n'.format(len(driver_list)))
    print(set(df['subject']))
    #subjects = list(set(df["subject"]))
    #select = random.sample(range(len(subjects)), 2)
    print()
    print('Split {} fold'.format(k))

    model_1_subjects = ['p041', 'p066', 'p016', 'p024', 'p056', 'p061']
    model_2_subjects = ['p075', 'p015', 'p042', 'p022', 'p049', 'p039']
    model_3_subjects = ['p035', 'p026', 'p072', 'p012', 'p047', 'p045']
    model_4_subjects = ['p081', 'p021', 'p050', 'p002', 'p064', 'p051']
    model_subjects = [model_1_subjects, model_2_subjects, model_3_subjects, model_4_subjects]

    valid_subjects = ['p014', 'p052']

    print('Model 1 train set:{}'.format(model_1_subjects))
    print('Model 2 train set:{}'.format(model_2_subjects))
    print('Model 3 train set:{}'.format(model_3_subjects))
    print('Model 4 train set:{}'.format(model_4_subjects))
    print('Valid set:{}'.format(valid_subjects))

    bagging_data_dir = data_dir + 'bagging/'
    if not os.path.exists(bagging_data_dir):
        os.mkdir(bagging_data_dir)

    for i in range(1, 5):
        if not os.path.exists(bagging_data_dir + "train{}".format(i)):
            os.mkdir(bagging_data_dir + "train{}".format(i))
            for j in range(10):
                os.mkdir(bagging_data_dir + "train{}".format(i) + "/c%d"%j)

    if not os.path.exists(bagging_data_dir + "valid"):
        os.mkdir(bagging_data_dir + "valid")
        for i in range(10):
            os.mkdir(bagging_data_dir + "valid/c%d"%i)

    for valid_subject in valid_subjects:
        df_valid = df[(df["subject"]==valid_subject)]
        for index, row in df_valid.iterrows():
            subpath = row["classname"] + "/" + row["img"]
            if os.path.exists(os.path.join(data_dir,"train",subpath)):
                shutil.move(os.path.join(data_dir,"train",subpath), os.path.join(bagging_data_dir,"valid",subpath),)
            else:
                print("cannot move {} : {}".format(row["subject"],subpath))


    for i, single_subjects in enumerate(model_subjects):
        for single_subject in single_subjects:
            df_train = df[(df["subject"]==single_subject)]
            for index, row in df_train.iterrows():
                subpath = row["classname"] + "/" + row["img"]
                if os.path.exists(os.path.join(data_dir,"train",subpath)):
                    target_dir = bagging_data_dir + "train{}".format(i+1)
                    shutil.move(os.path.join(data_dir,"train",subpath), os.path.join(target_dir,subpath),)
                else:
                    print("cannot move {} : {}".format(row["subject"],subpath))


#data_recover()
#splite_valid()
bagging_splite()