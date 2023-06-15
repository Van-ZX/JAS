#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
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
from YeNet import Yenet
from Xunet2 import XuNet
import scipy.io as scio


torch.cuda.empty_cache()

MAX_ZETA = 2.0
DELTA_ZETA0 = 0.01
INTER_ZETA = 1.0
DELTA_ZETA1 = 0.1

IMAGE_SIZE = 256
BATCH_SIZE = 1
payload = 0.3
OUTPUT_PATH = '/home/fanzexin/ye_beta0.1_adv_DejoinUpdist_payload' + str(payload)

msg = round(payload * IMAGE_SIZE * IMAGE_SIZE)
semi_msg = round(0.5 * payload * IMAGE_SIZE * IMAGE_SIZE)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



def test(model, device, loader):
    tst1 = 0
    tst2 = 0
    tst3 = 0
    tst4 = 0

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
        grad1 = tmp[0: :2, :]
        grad2 = tmp[1: :2, :]

        if not correct: # 能骗过分析器
            coefS = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            steg = Image.fromarray(coefS).convert('L')
            steg.save(os.path.join(OUTPUT_PATH, name[0]))
            tst1 = tst1 + 1
            continue

        if correct: # 分析器能准确分类
            cover = data[0, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            stego_ite = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            
            cost1 = scio.loadmat('/home/fanzexin/2dejoinUp_rho0.1_payload' + str(payload) + '/rho1_'+name[0].split('.')[-2]+'.mat')['cost1']
            cost2 = scio.loadmat('/home/fanzexin/2dejoinUp_rho0.1_payload' + str(payload) + '/rho2_'+name[0].split('.')[-2]+'.mat')['cost2']
            
            cover1 = cover[0: :2, :].reshape(1, -1).copy()
            cover2 = cover[1: :2, :].reshape(1, -1).copy()
            b_grad1 = np.zeros((2, IMAGE_SIZE * IMAGE_SIZE // 4))
            b_grad2 = np.zeros((2, IMAGE_SIZE * IMAGE_SIZE // 4))

            # import ipdb
            # ipdb.set_trace()

            shape1 = grad1.reshape(1, -1).copy()
            b_grad1[0, :] = shape1[:, 0: : 2]
            b_grad1[1, :] = shape1[:, 1: : 2]
            shape2 = grad2.reshape(1, -1).copy()
            b_grad2[0, :] = shape2[:, 0: : 2]
            b_grad2[1, :] = shape2[:, 1: : 2]

            GAMMA_MAX = 1
            DELTA_GAMMA = 0.1
            GAMMA = 0.0

            while GAMMA < GAMMA_MAX:
                GAMMA = GAMMA + DELTA_GAMMA
                if not correct:
                    break
                # import ipdb
                # ipdb.set_trace()
                b_rho2 = cost2.copy() # 9 * 16384
                rhonew2 = adjust_cost(b_rho2, b_grad2, 1 + GAMMA)
                b_steg2 = embed(cover2, rhonew2, semi_msg)
                stego_ite[1: :2, :] = b_steg2.reshape(128, 256).copy()

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
                    tst2 = tst2 + 1
                    break

            stego_ite = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)            
            GAMMA = 0.0
            while GAMMA < GAMMA_MAX:
                GAMMA = GAMMA + DELTA_GAMMA
                if not correct:
                    break
                
                b_rho1 = cost1.copy()
                rhonew1 = adjust_cost(b_rho1, b_grad1, 1 + GAMMA)
                b_steg1 = embed(cover1, rhonew1, semi_msg)
                stego_ite[0: :2, :] = b_steg1.reshape(128, 256).copy()

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
                    tst3 = tst3 + 1
                    break
            if correct:
                coefS = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
                steg = Image.fromarray(coefS).convert('L')
                steg.save(os.path.join(OUTPUT_PATH, name[0]))
                tst4 = tst4 + 1
    print("tst1 : " + str(tst1) + "\n")
    print("tst2 : " + str(tst2) + "\n")
    print("tst3 : " + str(tst3) + "\n")
    print("tst4 : " + str(tst4) + "\n")

            

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
        self.cover_dir = '/home/fanzexin/BOSSbase_1.01_BOWS2_256'
        self.stego_dir = '/home/fanzexin/2DejoinUpdist_beta0.1_payload' + str(payload)


        self.covers_list_all = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]
        
        
        # import ipdb
        # ipdb.set_trace()

        if (partition == 0):
            self.cover_list = self.covers_list_all[:]
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
        name = cover_path.split('/')[-1]
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
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    model = Yenet().cuda()
    model = nn.DataParallel(model)
    model.eval()

    all_state = torch.load(path)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
    test(model, device, loader)


if __name__ == '__main__':
    path = '/home/fanzexin/XuNet/yenet_boss_bows_hill_' + str(payload) + '_AUG/model_params.pt'

    main(path)