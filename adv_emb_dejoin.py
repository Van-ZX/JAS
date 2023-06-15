#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from curses.ascii import DEL
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
from adv_emb_cal import *
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

IMAGE_SIZE = 256
BATCH_SIZE = 1
OUTPUT_PATH = '/home/fanzexin/adv_emb_dejoin_0.4'
payload = 0.4


msg = round(payload * IMAGE_SIZE * IMAGE_SIZE)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def test(model, device, loader):

    for idx, sample in enumerate(loader):
        
        tst1 = 0
        tst2 = 0
        tst3 = 0

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

        # # compute sign of grad
        # criterion = nn.CrossEntropyLoss()
        # loss = criterion(output, label)
        # model.zero_grad()
        # loss.backward()
        # grad = data.grad.data
        # tmp = grad[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)


        if not correct: # 能骗过分析器
            tst1 = tst1 + 1
            coefS = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            steg = Image.fromarray(coefS).convert('L')
            steg.save(os.path.join(OUTPUT_PATH, name[0]))
            continue

        if correct: # 分析器能准确分类
            cover = data[0, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            stego_ite = data[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
            rhoini = jointrho(cover)
            rhoini = rhoini.reshape(9, IMAGE_SIZE, IMAGE_SIZE // 2)


            BETA_MAX = 1
            DELTA_BETA = 0.1
            BETA = 0.0
            img_len = IMAGE_SIZE*IMAGE_SIZE//2

            rand_ind = np.array(random.sample(range(img_len), k = img_len))
            
            while BETA < BETA_MAX + 0.01:
                BETA = BETA + DELTA_BETA
                if not correct:
                    break
                if BETA >= 0.999:
                    break
                

                conv_ind = rand_ind[0 : int((1-BETA)*img_len)]
                adv_ind = rand_ind[int((1-BETA)*img_len) : img_len]
                
                x_shape = cover.reshape(1, -1).copy()
                X1 = x_shape[:, 0: : 2]
                X2 = x_shape[:, 1: : 2]

                # b_grad = np.zeros((2, IMAGE_SIZE * IMAGE_SIZE // 2))
                # shape = tmp.reshape(1, -1).copy()
                # b_grad[0, :] = shape[:, 0: : 2]
                # b_grad[1, :] = shape[:, 1: : 2]
                b_rho = rhoini.reshape(9, -1).copy()

                b_cover = np.zeros((2, 256*128))
                b_cover[0, :] = X1.copy()
                b_cover[1, :] = X2.copy()
                b_steg = b_cover.copy()

                conv_cover = b_cover[:, conv_ind]
                adv_cover = b_cover[:, adv_ind]
                conv_rho = b_rho[:, conv_ind]
                adv_rho = b_rho[:, adv_ind]
                # if conv_cover.size == 0:
                #     import ipdb
                #     ipdb.set_trace()

                if BETA < 0.99:
                    conv_steg = embed(conv_cover, conv_rho, int(msg*(1-BETA)))
                    b_steg[:, conv_ind] = conv_steg

                ttt = np.zeros((1, 256*256))
                ttt[:, 0 : : 2] = b_steg[0, :].copy()
                ttt[:, 1 : : 2] = b_steg[1, :].copy()

    

                newdata = np.zeros((2, stego_ite.shape[0], stego_ite.shape[1]))
                newdata[0, :, :] = cover.copy()
                newdata[1, :, :] = ttt.reshape(256, 256).copy()
                newdata = torch.from_numpy(newdata).reshape((2, 1, stego_ite.shape[0], stego_ite.shape[1]))
                newdata = newdata.to(device)
                newdata.requires_grad = True
                newdata = newdata.float()
                newdata.retain_grad()

                output = model(newdata)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, label)
                model.zero_grad()
                loss.backward()
                grad = newdata.grad.data
                tmp = grad[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)

                b_grad = np.zeros((2, IMAGE_SIZE * IMAGE_SIZE // 2))
                shape = tmp.reshape(1, -1).copy()
                b_grad[0, :] = shape[:, 0: : 2]
                b_grad[1, :] = shape[:, 1: : 2]
                adv_grad = np.zeros((2, stego_ite.shape[0], stego_ite.shape[1]))
                adv_grad = b_grad[:, adv_ind]

                rhonew = adjust_cost(adv_rho, adv_grad, 2)
                b_steg[:, adv_ind] = embed(adv_cover, rhonew, int(msg*BETA))
                # stego_ite = b_steg.copy()
                
                ttt = np.zeros((1, 256*256))
                ttt[:, 0 : : 2] = b_steg[0, :].copy()
                ttt[:, 1 : : 2] = b_steg[1, :].copy()

                newdata = np.zeros((2, stego_ite.shape[0], stego_ite.shape[1]))
                newdata[0, :, :] = cover.copy()
                newdata[1, :, :] = ttt.reshape(256, 256).copy()
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
                    tst2 = tst2 + 1
                    steg = Image.fromarray(stego_ite).convert('L')
                    steg.save(os.path.join(OUTPUT_PATH, name[0]))
                    # with open('victory_max.txt', 'a') as file:
                    #     file.write(name[0] + '\n')
                    break

            if correct:
                tst3 = tst3 + 1
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
        self.cover_dir = '/home/fanzexin/BOSSbase_1.01_BOWS2_256'
        self.stego_dir = '/home/fanzexin/HILL_payload0.2'


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
    model = XuNet().cuda()
    model = nn.DataParallel(model)
    model.eval()

    all_state = torch.load(path)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
    test(model, device, loader)


if __name__ == '__main__':
    path = '/home/fanzexin/XuNet/stego_hill_boss_bow_0.4_AUG/model_params.pt'

    main(path)