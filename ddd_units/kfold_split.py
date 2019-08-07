# coding:utf-8

import os
import json
import fire
import pandas as pd
import numpy as np
import shutil
import random
import inspect
from glob import glob

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def kfold_dataset(data_dir, number_valids=2, number_models=4, 
                  output_filename='kfold_bagging.json'):
    data_dir = os.path.abspath(data_dir)
    driver_imgs_list_csv = os.path.join(data_dir, "driver_imgs_list.csv")
    df = pd.read_csv(driver_imgs_list_csv)
    drivers = list(set(df["subject"]))

    k_valid_index_drivers = []
    def select_valid_drivers():
        selects = random.sample(range(len(drivers)), 2)
        for i in selects:
            if i in k_valid_index_drivers:
                return None
        k_valid_index_drivers.extend(selects)
        select_drivers = [drivers[int(s)] for s in selects]
        return select_drivers

    valid_drivers = {}
    select_drivers = None
    for i in range(number_models):
        select_drivers = None
        # select valid set
        while True:
            select_drivers = select_valid_drivers()
            if select_drivers is None:
                print('select drivers has already in valid drivers, replay again')
                continue
            break
        print('{}th select valid drivers = {}'.format(i+1, select_drivers))
        valid_drivers['model_{}'.format(i+1)] = select_drivers

    # write config file
    config_path = os.path.join(
        data_dir, output_filename)
    with open(config_path, "w") as f:
        state_dict = {
            'number_models': number_models,
            'number_valids': number_valids,
            'valid_drivers': valid_drivers
        }
        json.dump(state_dict, f)
    print('save config file to "{}"'.format(config_path))


if __name__ == '__main__':
    fire.Fire()