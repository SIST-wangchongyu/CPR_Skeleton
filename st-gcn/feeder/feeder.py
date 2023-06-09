# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
# visualization
import time

# operation
# import tools
from . import tools


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the input sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)
        self.class_sample_count = np.unique(self.label, return_counts=True)[1]
        self.weight = 1. / self.class_sample_count
        self.samples_weight = self.weight[self.label]
        print(self.samples_weight)
        self.sampler = WeightedRandomSampler(self.samples_weight, len(self.label))

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        print(self.label_path)
        self.label = np.load(self.label_path)

        # load data

        self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # print(data_numpy.shape)
        # print(label.shape)
        return data_numpy.astype(np.float32), label

def feeder_data_generator(dataset, batch_size,sampler):
    if sampler == 1:
        data_loader = torch.utils.data.DataLoader(dataset, sampler=dataset.sampler, batch_size=batch_size,
                                                  num_workers=2,
                                                  pin_memory=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size,
                                                  num_workers=2,
                                                  pin_memory=True)

    return data_loader
if __name__ == '__main__':
    data_path = '/public/home/wangchy5/CPR/st-gcn/resource/Processed_Data/train.npy'
    label_path = '/public/home/wangchy5/CPR/st-gcn/resource/Processed_Data/train_label.npy'
    skeleton = Feeder(data_path,label_path)
    data_loader = torch.utils.data.DataLoader(skeleton, batch_size=1,
                                                  pin_memory=True)
    process =tqdm(data_loader)
    for batch_idx, (data, target) in enumerate(process):
        print(data)
        print(target)


