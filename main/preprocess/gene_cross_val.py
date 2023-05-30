import os
import copy
import random
import itertools
import numpy as np


def split_list(my_list, num_parts):
    part_size = len(my_list) // num_parts
    remainder = len(my_list) % num_parts
    result = []
    start = 0
    for i in range(num_parts):
        if i < remainder:
            end = start + part_size + 1
        else:
            end = start + part_size
        result.append(my_list[start:end])
        start = end
    return result

def build_dataset(lab_path, save_dir, num_folds=5, num_classes=7):
    os.makedirs(save_dir, exist_ok=True)
    lab_data = np.loadtxt(lab_path, dtype=np.str_)
    n_fold_list = []
    for cls_idx in range(num_classes):
        data_list = []
        for data in lab_data:
            if int(data[-1]) == cls_idx:
                data_list.append(data[0])
        random.shuffle(data_list)
        data_info_list = []
        for item in data_list:
            data_info_list.append(
                {'data': item,
                 'cls': cls_idx, })
        n_fold_list.append(split_list(data_info_list, num_folds))
    for i in range(num_folds):
        train_data = []
        val_data = []
        _n_fold_list = copy.deepcopy(n_fold_list)
        for j in range(num_classes):
            val_data.append(_n_fold_list[j][i])
            del _n_fold_list[j][i]
            for item in _n_fold_list[j]:
                train_data.append(item)
        train_data = list(itertools.chain(*train_data))
        val_data = list(itertools.chain(*val_data))
        print(f'Fold {i+1}: num_train_data={len(train_data)}, num_val_data={len(val_data)}')
        train_file = open(f'{save_dir}/train_fold{i+1}.txt', 'w')
        val_file = open(f'{save_dir}/val_fold{i+1}.txt', 'w')
        for item in train_data:
            data = item['data']
            cls = item['cls']
            train_file.write(f'{data} {cls}'+'\n')
        for item in val_data:
            data = item['data']
            cls = item['cls']
            val_file.write(f'{data} {cls}'+'\n')
        train_file.close()
        val_file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data preprocessing Config', add_help=False)
    parser.add_argument('--lab-path', default='', type=str)
    parser.add_argument('--save-dir', default='', type=str)
    parser.add_argument('--num-folds', default=5, type=int)
    parser.add_argument('--seed', default=66, type=int)
    args = parser.parse_args()
    random.seed(args.seed) ## Adjust according to model performance
    # lab_path = 'data/classification_dataset/labels/labels.txt'
    # save_dir = 'data/classification_dataset/labels/'
    build_dataset(args.lab_path, args.save_dir, num_folds=args.num_folds)
