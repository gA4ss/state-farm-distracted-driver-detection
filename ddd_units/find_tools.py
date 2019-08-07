# coding:utf-8

import os
import json
import shutil
from glob import glob

def sort_models(models_path, evaluate_type='accuracy'):
    models_path = os.path.abspath(models_path)
    checkpoint_paths = glob(os.path.join(models_path, "*.json"))

    results = []
    for checkpoint_path in checkpoint_paths:
        with open(checkpoint_path, 'r') as load_f:
            config = json.load(load_f)
            last_accuracy = config['best_valid_accuracy']
            last_loss = config['best_valid_loss']

            if evaluate_type == 'accuracy':
                results.append((last_accuracy, checkpoint_path))
            else:
                results.append((last_loss, checkpoint_path))
    def take_evaluate(elem):
        return elem[0]
    
    if evaluate_type == 'accuracy':
        reverse=True
    else:
        reverse=False

    results.sort(key=take_evaluate, reverse=reverse)
    return results

def copy_incorrect_images(data_dir, incorrect_path, output_dir):
    data_dir = os.path.abspath(data_dir)
    incorrect_path = os.path.abspath(incorrect_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(incorrect_path, 'r') as load_f:
        for line in load_f:
            records = line.split(':')
            image_name = records[0]
            class_dir = 'c{}'.format(records[1])
            image_path = os.path.join(data_dir, 'train', class_dir, image_name)
            output_path = os.path.join(output_dir, class_dir)

            if not os.path.exists(output_path):
                os.mkdir(output_path)

            output_path = os.path.join(output_path, image_name)
            if os.path.exists(image_path):
                shutil.copy(image_path, output_path)
                print('copy "{}" to "{}" successed'.format(image_path, output_path))
            else:
                print("cannot copy {} : {}".format(image_path, output_path))

if __name__ == '__main__':
    results = sort_models('../model', evaluate_type='loss')
    for result in results:
        print('{} --- {}'.format(result[0], result[1]))
    #copy_incorrect_images('../data', './incorrects.log', './incorrect_images')
 