#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
import os
import logging
import numpy as np
import pandas as pd
from YeNet import Yenet
from srm_filter_kernel import *
from MPNCOV import *
import cv2
from pathlib import Path
import copy
import random
from glob import glob
from cmd_cost import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from Xunet2 import XuNet
import scipy.io as scio
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.empty_cache()

MAX_ZETA = 2.0
DELTA_ZETA0 = 0.01
INTER_ZETA = 1.0
DELTA_ZETA1 = 0.1

count_sub = 4 # 分块数量

IMAGE_SIZE = 256
BATCH_SIZE = 1
OUTPUT_PATH = '/home/fanzexin/xu_adv_cmd_0.2'
payload = 0.2

msg = round(payload * IMAGE_SIZE * IMAGE_SIZE / count_sub)
qua_msg = round(payload * IMAGE_SIZE * IMAGE_SIZE / (4*count_sub))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def test(model, device, loader):
    tst1 = 0
    tst2 = 0
    tst3 = 0

    for idx, sample in enumerate(loader):

        data, label, name = sample['data'], sample['label'], sample['name']
        # name = np.str(name.data[0, 0]) + '.pgm'
        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)

        data, label = data.cuda(), label.cuda()
        data.requires_grad = True
        output = model(data)  # FP
        pred = output.max(1, keepdim=True)[1]
        # correct = pred.eq(label.view_as(pred)).sum().item()
        correct = bool(pred[1, 0].data)

        if not correct: # 能骗过分析器
            tst1 = tst1 + 1
            coefS = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            steg = Image.fromarray(coefS).convert('L')
            steg.save(os.path.join(OUTPUT_PATH, name[0]))
            continue

        if correct: # 分析器能准确分类
            # import ipdb
            # ipdb.set_trace()

            cover = data[0, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            coefZ = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            rhop_mat_name = '/home/fanzexin/CMDrho' + str(payload) + '/rhoP_' + name[0].split('.')[-2] + '.mat'
            rhom_mat_name = '/home/fanzexin/CMDrho' + str(payload) + '/rhoM_' + name[0].split('.')[-2] + '.mat'
            prhoini = scio.loadmat(rhop_mat_name)['rhoP']
            mrhoini = scio.loadmat(rhom_mat_name)['rhoM']

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, label)
            model.zero_grad()
            loss.backward()
            grad = data.grad.data
            tmp = grad[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)

            DELTA_GAMMA = 0.1
            MAX_GAMMA = 1.0


            for i in range(count_sub // 2):
                if not correct:
                        break
                for j in range(count_sub // 2):
                    if not correct:
                        break
                    b_cover = cover[i: :2, j: :2].copy()                    
                    pb_rho = prhoini[i: :2, j: :2].copy()
                    mb_rho = mrhoini[i: :2, j: :2].copy()
                    sign_grad = tmp[i: :2, j: :2]
                    # import ipdb
                    # ipdb.set_trace()
                    GAMMA = 0.0
                    while GAMMA < MAX_GAMMA:
                        # import ipdb
                        # ipdb.set_trace()

                        GAMMA = GAMMA + DELTA_GAMMA
                        pb_rho[sign_grad<0] *= (1+GAMMA)
                        pb_rho[sign_grad>0] /= (1+GAMMA)
                        mb_rho[sign_grad>0] *= (1+GAMMA)
                        mb_rho[sign_grad<0] /= (1+GAMMA)
                        b_steg = hill_emb(b_cover, pb_rho, mb_rho, qua_msg)
                        coefZ[i: :2, j: :2] = b_steg.copy()

                        newdata = np.zeros((2, coefZ.shape[0], coefZ.shape[1]))
                        newdata[0, :, :] = cover.copy()
                        newdata[1, :, :] = coefZ.copy()
                        newdata = torch.from_numpy(newdata).reshape((2, 1, coefZ.shape[0], coefZ.shape[1]))
                        newdata = newdata.cuda()
                        newdata.requires_grad = True
                        newdata = newdata.float()
                        newdata.retain_grad()
                        output = model(newdata)  # FP
                        pred = output.max(1, keepdim=True)[1]
                        correct = bool(pred[1, 0].data)
                        if not correct:
                            # import ipdb
                            # ipdb.set_trace()
                            tst2 = tst2 + 1
                            steg = Image.fromarray(coefZ).convert('L')
                            steg.save(os.path.join(OUTPUT_PATH, name[0]))
                            break
                        
                    coefZ = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
                    if correct and i == 1 and j == 1:
                        tst3 = tst3 + 1
                        coefS = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
                        steg = Image.fromarray(coefS).convert('L')
                        steg.save(os.path.join(OUTPUT_PATH, name[0]))
    print("tst1 : " + str(tst1) + "\n")
    print("tst2 : " + str(tst2) + "\n")
    print("tst3 : " + str(tst3) + "\n")

            

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
        self.stego_dir = '/home/fanzexin/stego_BOSS_BOWS_HILL_4CMD_payload0.2'


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
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    # model = Yenet().cuda()
    model = XuNet().cuda()
    model = nn.DataParallel(model)
    model.eval()

    all_state = torch.load(path)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
    test(model, device, loader)


if __name__ == '__main__':
    # path = '/home/fanzexin/XuNet/yenet_boss_bows_hill_0.3_AUG/model_params.pt'
    path = '/home/fanzexin/XuNet/stego_hill_boss_bow_0.2/model_params.pt'

    main(path)