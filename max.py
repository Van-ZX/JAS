#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
import os
import logging
import numpy as np
import pandas as pd
from srm_filter_kernel import *
from MPNCOV import *
import cv2
from pathlib import Path
import copy
import random
from glob import glob
from calcost import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from Xunet2 import XuNet

torch.cuda.empty_cache()

MAX_ZETA = 2.0
DELTA_ZETA0 = 0.01
INTER_ZETA = 1.0
DELTA_ZETA1 = 0.1

count_sub = 4 # 分块数量

IMAGE_SIZE = 512
BATCH_SIZE = 1
OUTPUT_PATH = 'hill_3.0_0.4'
payload = 0.4

msg = round(payload * IMAGE_SIZE * IMAGE_SIZE / count_sub)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def test(model, device, loader):

    for idx, sample in enumerate(loader):

        data, label, name = sample['data'], sample['label'], sample['name']
        # name = np.str(name.data[0, 0]) + '.pgm'
        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)

        data, label = data.to(device), label.to(device)
        data.requires_grad = True
        output = model(data)  # FP
        pred = output.max(1, keepdim=True)[1]
        # correct = pred.eq(label.view_as(pred)).sum().item()
        correct = bool(pred[1, 0].data)

        # compute sign of grad
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        model.zero_grad()
        loss.backward()
        grad = data.grad.data
        tmp = grad[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)

        # import ipdb
        # ipdb.set_trace()

        if not correct: # 能骗过分析器
            coefS = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            steg = Image.fromarray(coefS).convert('L')
            steg.save(os.path.join(OUTPUT_PATH, name[0]))
            continue

        if correct: # 分析器能准确分类
            cover = data[0, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            stego_ite = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            rhoini = jointrho(cover)
            rhoini = rhoini.reshape(9, IMAGE_SIZE, IMAGE_SIZE // 2)
            
            # stego_ite = np.zeros((data.shape[2], data.shape[3]))
            # import ipdb
            # ipdb.set_trace()

            for i in range(count_sub // 2):
                for j in range(count_sub // 2):
                    b_cover = cover[IMAGE_SIZE*i*2//count_sub : IMAGE_SIZE*(i+1)*2//count_sub, IMAGE_SIZE*j*2//count_sub : IMAGE_SIZE*(j+1)*2//count_sub]
                    b_rho = rhoini[:, IMAGE_SIZE*i*2//count_sub : IMAGE_SIZE*(i+1)*2//count_sub, IMAGE_SIZE*j//count_sub : IMAGE_SIZE*(j+1)//count_sub]
                    b_steg = embed(b_cover, b_rho, msg)
                    stego_ite[IMAGE_SIZE*i*2//count_sub : IMAGE_SIZE*(i+1)*2//count_sub, IMAGE_SIZE*j*2//count_sub : IMAGE_SIZE*(j+1)*2//count_sub] = b_steg

            ALPHA = 3.0   


            # if ZETA > INTER_ZETA:
            #     DELTA_ZETA = DELTA_ZETA1
            # ZETA += DELTA_ZETA
            for i in range(count_sub // 2):
                for j in range(count_sub // 2):
                    b_cover = cover[IMAGE_SIZE*i*2//count_sub : IMAGE_SIZE*(i+1)*2//count_sub, IMAGE_SIZE*j*2//count_sub : IMAGE_SIZE*(j+1)*2//count_sub]
                    b_grad = np.zeros((2, IMAGE_SIZE * IMAGE_SIZE // 8))
                    shape = tmp[IMAGE_SIZE*i*2//count_sub : IMAGE_SIZE*(i+1)*2//count_sub, IMAGE_SIZE*j*2//count_sub : IMAGE_SIZE*(j+1)*2//count_sub].reshape(1, -1).copy()
                    b_grad[0, :] = shape[:, 0: : 2]
                    b_grad[1, :] = shape[:, 1: : 2]
                    b_rho = rhoini[:, IMAGE_SIZE*i*2//count_sub : IMAGE_SIZE*(i+1)*2//count_sub, IMAGE_SIZE*j//count_sub : IMAGE_SIZE*(j+1)//count_sub].reshape(9, -1).copy()
                    # import ipdb
                    # ipdb.set_trace()

                    rhonew = adjust_cost(b_rho, b_grad, ALPHA)
                    b_steg = embed(b_cover, rhonew, msg)
                    stego_ite[IMAGE_SIZE*i*2//count_sub : IMAGE_SIZE*(i+1)*2//count_sub, IMAGE_SIZE*j*2//count_sub : IMAGE_SIZE*(j+1)*2//count_sub] = b_steg


                    newdata = np.zeros((2, stego_ite.shape[0], stego_ite.shape[1]))
                    newdata[0, :, :] = cover.copy()
                    newdata[1, :, :] = stego_ite.copy()
                    newdata = torch.from_numpy(newdata).reshape((2, 1, stego_ite.shape[0], stego_ite.shape[1]))
                    newdata = newdata.to(device)
                    newdata.requires_grad = True
                    newdata = newdata.float()
                    newdata.retain_grad()

                    output = model(newdata)  # FP
                    pred = output.max(1, keepdim=True)[1]
                    correct = bool(pred[1, 0].data)

                    if not correct:
                        # import ipdb
                        # ipdb.set_trace()
                        
                        steg = Image.fromarray(stego_ite).convert('L')
                        steg.save(os.path.join(OUTPUT_PATH, name[0]))
                        # with open('victory_max.txt', 'a') as file:
                        #     file.write(name[0] + '\n')
                        break

                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(output, label)
                    model.zero_grad()
                    loss.backward()
                    grd = newdata.grad.data
                    tmp = grd[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
                    # sgn = np.zeros((2, IMAGE_SIZE * IMAGE_SIZE // 2))
                    # shape = tmp.reshape(1, -1).copy()
                    # sgn[0, :] = shape[:, 0: : 2]
                    # sgn[1, :] = shape[:, 1: : 2]

                    if correct and i == 1 and j == 1:
                        coefS = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
                        steg = Image.fromarray(coefS).convert('L')
                        steg.save(os.path.join(OUTPUT_PATH, name[0]))

            

class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


class MyDataset(Dataset):
    def __init__(self, partition, transform):
        random.seed(1234)

        self.transform = transform
        # self.cover_dir = '/public/fanzexin/adv_dejoin/cover'
        self.cover_dir = '/data-x/g16/BossBase512'
        self.stego_dir = '/public/fanzexin/dejoin_matlab/stego_HILL_payload0.4'


        self.covers_list_all = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]
        
        
        # import ipdb
        # ipdb.set_trace()

        if (partition == 0):
            self.cover_list = self.covers_list_all[:10000]
            self.cover_paths= [os.path.join(self.cover_dir, x) for x in  self.cover_list]
            self.cover_paths = self.cover_paths 
            self.stego_paths = [os.path.join(self.stego_dir, x) for x in self.cover_list]
            self.stego_paths = self.stego_paths 
            self.cover_steg = list(zip(self.cover_paths, self.stego_paths))
            # random.shuffle(self.cover_steg)
            self.cover_paths, self.stego_paths = zip(*self.cover_steg)

        assert len(self.cover_paths) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_paths)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = self.cover_paths[file_index]
        stego_path = self.stego_paths[file_index]
        cover_data = cv2.imread(cover_path, -1)
        stego_data = cv2.imread(stego_path, -1)
        # import ipdb
        # ipdb.set_trace()

        # name = int(cover_path[24:-4])
        # name = np.array([1], dtype=np.int)*name
        name = str(cover_path[24:])
        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')
        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = name
        return sample


def main(path):
    device = torch.device("cuda")
    kwargs = {'num_workers': 4, 'pin_memory': False}
    train_transform = transforms.Compose([
        ToTensor()
    ])

    train_dataset = MyDataset(0, train_transform)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    model = XuNet().cuda()
    model = nn.DataParallel(model)
    model.eval()

    all_state = torch.load(path)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
    test(model, device, loader)


if __name__ == '__main__':
    path = '/public/fanzexin/XuNet/stego_hill_0.4_AUG/model_params.pt'

    main(path)