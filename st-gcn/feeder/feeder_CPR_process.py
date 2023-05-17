# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
from torchvision import datasets
from tqdm import tqdm
# operation
# from . import tools


class Feeder_CPR(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=5,
                 num_person_out=2,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        label_path = self.label_path
        self.label = np.load(label_path)
        self.data  = np.load(data_path)
        # print(self.data.shape)
        # output data shape (N, C, T, V, M)
          #sample
        self.C = 3  #channel
        self.T = 64  #frame
        self.V = 13 #joint
        self.M = 1  #person

    def __len__(self):
        return (self.label.shape[0])

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        data_input = self.data[index]
        label_input      =self.label[index]
        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.M))
        for i  in range(data_input.shape[2]):
            frame_index = i
            data_numpy[0, frame_index, :, 0] = data_input[0,i]
            data_numpy[1, frame_index, :, 0] = data_input[1,i]
            data_numpy[2, frame_index, :, 0] = data_input[2,i]

        # centralization
        # data_numpy[0:2] = data_numpy[0:2] - 0.5
        # data_numpy[0][data_numpy[2] == 0] = 0
        # data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        

        return data_numpy, label_input

    # def top_k(self, score, top_k):
    #     assert (all(self.label >= 0))
    #
    #     rank = score.argsort()
    #     hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
    #     return sum(hit_top_k) * 1.0 / len(hit_top_k)
    #
    # def top_k_by_category(self, score, top_k):
    #     assert (all(self.label >= 0))
    #     return tools.top_k_by_category(self.label, score, top_k)
    #
    # def calculate_recall_precision(self, score):
    #     assert (all(self.label >= 0))
    #     return tools.calculate_recall_precision(self.label, score)
if __name__ == '__main__':
    data_path = '/public/home/wangchy5/CPR/st-gcn/resource/Data_Skeleton/train.npy'
    label_path = '/public/home/wangchy5/CPR/st-gcn/resource/Data_Skeleton/train_label.npy'
    skeleton = Feeder_CPR(data_path,label_path)
    data_loader = torch.utils.data.DataLoader(skeleton, batch_size=1,
                                                  pin_memory=True)
    process =tqdm(data_loader)
    for batch_idx, (data, target) in enumerate(process):
        print(batch_idx)
        if batch_idx ==0:
            data_processed = data
            label_processed = target
        else:
            data_processed = torch.cat((data_processed,data),dim=0)
            label_processed = torch.cat((label_processed,target),dim=0)

        print(data_processed.shape )
    print(data_processed.shape)
    print(label_processed.shape)
