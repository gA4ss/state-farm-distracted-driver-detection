# coding:utf-8
import os
import re
import fire
from glob import glob
import matplotlib.pyplot as plt

def plot_log_dir(log_dir, save_fig=True, show_fig=False):
    log_files = glob(os.path.join(log_dir, "*.log"))
    for log_file in log_files:
        plot_log(log_file, save_fig=save_fig, show_fig=show_fig)

def plot_log(log_file, save_fig=True, show_fig=False):
    log_file = os.path.abspath(log_file)
    losses = []
    accuracies = []

    with open(log_file, 'r') as f:
        line = f.readline()
        while line:
            #print(line)
            if line[0:5] == 'Train' and 'Trainning' in line:
                epoch = int(line[11:])
            elif line[0:5] == 'Valid':
                loss = re.search(r'[0-9]+\.[0-9]*', line)
                loss = float(loss.group())
                losses.append(loss)
                equ = re.search(r'[0-9]+/[0-9]+', line)
                equ = equ.group()
                a = ''
                b = ''
                change = False
                for e in equ:
                    if e == '/':
                        change = True
                        continue
                    
                    if change is False:
                        a += e
                    else:
                        b += e
                a = int(a)
                b = int(b)
                accuracies.append(a/b)
                print('epoch: {}, loss = {}, accuracy = {}'.format(epoch, loss, a/b))
            line = f.readline()
        f.close()
    # plot
    if len(losses) == 100 and len(accuracies) == 100:
        plt.suptitle('Valid set: Average of loss and accuracy')
        plt.figure(figsize=(9, 3))
        plt.subplot(1,2,1)
        plt.ylabel('loss')
        plt.plot(losses, 'r--')

        plt.subplot(1,2,2)
        plt.ylabel('accuracy')
        plt.plot(accuracies, 'b--')
        basename = os.path.basename(log_file)
        basename = basename[0:basename.find('.')]
        figname = os.path.join(os.path.dirname(log_file), basename+'.png')
        if save_fig:
            print('save to "{}"'.format(figname))
            plt.savefig(figname)

        if show_fig:
            plt.show()
    else:
        print('[ERROR]read log file failed.')

if __name__ == '__main__':
    #plot_log_dir('./log/')
    #plot_log('./log/xception4_Thu_Jul__4_03_28_18_2019.log')
    fire.Fire()


