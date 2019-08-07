# coding:utf-8

import os
import inspect
import pandas as pd
import matplotlib.pyplot as plt


def plot_driver_status() :
    data_dir = os.path.abspath(os.path.join(inspect.getfile(plot_driver_status), os.pardir))
    driver_imgs_list_csv = os.path.join(data_dir, "../data/driver_imgs_list.csv")
    df = pd.read_csv(driver_imgs_list_csv)
    df['classname'].value_counts().plot.bar()
    plt.show()


if __name__ == '__main__':
    plot_driver_status()