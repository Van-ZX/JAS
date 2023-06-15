#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
import os
from scipy import signal
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
# from calcost import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from Xunet2 import XuNet
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.empty_cache()

MAX_ZETA = 2.0
DELTA_ZETA0 = 0.01
INTER_ZETA = 1.0
DELTA_ZETA1 = 0.1

CMD_FACTOR = 10
MAX_ETA = 5

IMAGE_SIZE = 256
BATCH_SIZE = 1
OUTPUT_PATH = '/home/fanzexin/noCMDpre_0.4'
payload = 0.4

msg = round(payload * IMAGE_SIZE * IMAGE_SIZE)
len_block_msg = round(payload * IMAGE_SIZE * IMAGE_SIZE / 4)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def test(model, loader):

    for idx, sample in enumerate(loader):

        data, label, name = sample['data'], sample['label'], sample['name']
        # name = np.str(name.data[0, 0]) + '.pgm'
        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)
        
        # import ipdb
        # ipdb.set_trace()
        data, label = data.cuda(), label.cuda()
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
        grd = grad[1, :, :].cpu().detach().numpy().copy().reshape(1, IMAGE_SIZE*IMAGE_SIZE)


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
            rhoP1, rhoM1 = cost_hill(cover)
            rhoP1N = rhoP1.copy()
            rhoM1N = rhoM1.copy()
            coefS = cover.copy()
            ALPHA = 1  # ALPHA = 1+ETA
            b_index = random.randint(0, 3)
            bs_index = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int)

            for i in range(4):
                b_index = (b_index + 1) % 4
                r_0 = bs_index[b_index, 0]
                c_0 = bs_index[b_index, 1]
                rhoP1b = rhoP1N[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2]
                rhoM1b = rhoM1N[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2]
                b_block = EmbeddingSimulator(rhoP1b, rhoM1b, len_block_msg)
                coefS[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2] = coefS[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2] + b_block
                c_cmd = neighbor_change(cover, coefS)
                rhoP1N = rhoP1.copy()
                rhoM1N = rhoM1.copy()
                rhoP1N[c_cmd >= 1] = rhoP1N[c_cmd >= 1] / CMD_FACTOR
                rhoM1N[c_cmd <= -1] = rhoM1N[c_cmd <= -1] / CMD_FACTOR

            for i in range(4):
                if not correct:
                    break

                r_0 = bs_index[b_index, 0]
                c_0 = bs_index[b_index, 1]
                # initialize the block values.
                b_cover = cover[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2]
                b_rhoP1 = rhoP1N[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2]
                b_rhoM1 = rhoM1N[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2]
                # resume pending for perturbing to cover.
                stego_1 = coefS.copy()
                stego_1[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2] = b_cover
                
                newdata = np.zeros((2, stego_ite.shape[0], stego_ite.shape[1]))
                newdata[0, :, :] = cover.copy()
                newdata[1, :, :] = stego_1.copy()
                newdata = torch.from_numpy(newdata).reshape((2, 1, stego_ite.shape[0], stego_ite.shape[1]))
                newdata = newdata.to(device)
                newdata.requires_grad = True
                newdata = newdata.float()
                newdata.retain_grad()

                output = model(newdata)  # FP
                pred = output.max(1, keepdim=True)[1]
                correct = bool(pred[1, 0].data)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, label)
                model.zero_grad()
                loss.backward()
                grad = data.grad.data
                grad_s = grad[1, :, :].cpu().detach().numpy().copy().reshape(IMAGE_SIZE, IMAGE_SIZE)
                b_sign = np.sign(grad_s[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2])
                for ETA in np.arange(0.1, 1, 0.1):
                    rhoP1b = b_rhoP1.copy()
                    rhoM1b = b_rhoM1.copy()
                    ALPHA = 1 + ETA
                    rhoP1b[b_sign == 1] = rhoP1b[b_sign == 1] / ALPHA
                    rhoM1b[b_sign == 1] = rhoM1b[b_sign == 1] * ALPHA
                    rhoP1b[b_sign == -1] = rhoP1b[b_sign == -1] * ALPHA
                    rhoM1b[b_sign == -1] = rhoM1b[b_sign == -1] / ALPHA
                    b_block = EmbeddingSimulator(rhoP1b, rhoM1b, len_block_msg)
                    b_stego = b_cover + b_block
                    stego_1 = coefS.copy()
                    stego_1[r_0:IMAGE_SIZE:2, c_0:IMAGE_SIZE:2] = b_stego
                    
                    newdata = np.zeros((2, stego_ite.shape[0], stego_ite.shape[1]))
                    newdata[0, :, :] = cover.copy()
                    newdata[1, :, :] = stego_1.copy()
                    newdata = torch.from_numpy(newdata).reshape((2, 1, stego_ite.shape[0], stego_ite.shape[1]))
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
                        
                        steg = Image.fromarray(stego_1).convert('L')
                        steg.save(os.path.join(OUTPUT_PATH, name[0]))
                        # with open('victory_max.txt', 'a') as file:
                        #     file.write(name[0] + '\n')
                        break

                b_index = (b_index+1) % 4
         
            if correct:
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
        self.stego_dir = '/home/fanzexin/HILL_payload0.4'


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
    # device = torch.device("cuda:0")
    kwargs = {'num_workers': 4, 'pin_memory': False}
    train_transform = transforms.Compose([
        ToTensor()
    ])

    train_dataset = MyDataset(0, train_transform)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    # import ipdb
    # ipdb.set_trace()

    model = XuNet().cuda()
    model = nn.DataParallel(model)
    model.eval()
    
    all_state = torch.load(path)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
    test(model, loader)

def neighbor_change(cover, stego):
    h = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float)
    d = stego.astype(np.float32) - cover.astype(np.float32)
    c = signal.convolve2d(d, h, boundary='symm', mode='same')
    return c

def EmbeddingSimulator(rhoP, rhoM, m):
    n = rhoP.size
    Lambda = calc_lambda(rhoP, rhoM, m, n)

    zp = np.exp(-Lambda*rhoP)
    zm = np.exp(-Lambda*rhoM)
    z0 = 1+zp+zm
    pChangeP1 = zp/z0
    pChangeM1 = zm/z0
    
    randChange = np.random.rand(rhoP.shape[0], rhoP.shape[1])
    modification = np.zeros([rhoP.shape[0], rhoP.shape[1]])
    modification[randChange < pChangeP1] = 1
    modification[randChange >= 1-pChangeM1] = -1
    return modification

def cost_hill(X):
	wetCost = 1e10
	hp = np.array([[-0.25, 0.5, -0.25], [0.5, -1, 0.5], [-0.25, 0.5, -0.25]], dtype=np.float32)
	r_1 = signal.convolve2d(X.astype(np.float32), hp, boundary='symm', mode='same')
	lp_1 = np.ones([3, 3], dtype=np.float32)/9
	r_2 = signal.convolve2d(np.abs(r_1), lp_1, boundary='symm', mode='same')
	rho = 1/(r_2+1e-10)
	lp_2 = np.ones([15, 15], dtype=np.float32)/225
	rho = signal.convolve2d(rho, lp_2, boundary='symm', mode='same')
	rho[rho > 50] = wetCost
	rho[np.isnan(rho)] = wetCost
	rho_p = rho.copy()
	rho_m = rho.copy()
	rho_p[X == 255] = wetCost
	rho_m[X == 0] = wetCost
	return rho_p, rho_m

def calc_lambda(rhoP, rhoM, message_length, n):
	l3 = 1e+3
	m3 = message_length+1
	iterations = 0
	while m3 > message_length:
		l3 = l3*2
		zp = np.exp(-l3*rhoP)
		zm = np.exp(-l3*rhoM)
		z0 = 1+zp+zm
		pP1 = zp/z0
		pM1 = zm/z0
		m3 = ternary_entropyf(pP1, pM1)
		iterations = iterations+1
		if iterations > 10:
			Lambda = l3
			return Lambda

	l1 = 0
	m1 = n
	Lambda = 0

	alpha = float(message_length)/n
	while (float(m1-m3)/n > alpha/1000.0) and (iterations < 30):
		Lambda = l1+(l3-l1)/2.0
		zp = np.exp(-Lambda*rhoP)
		zm = np.exp(-Lambda*rhoM)
		z0 = 1+zp+zm
		pP1 = zp/z0
		pM1 = zm/z0
		m2 = ternary_entropyf(pP1, pM1)
		if m2 < message_length:
			l3 = Lambda
			m3 = m2
		else:
			l1 = Lambda
			m1 = m2
		iterations = iterations+1
	return Lambda


def ternary_entropyf(pP1, pM1):
	p0 = 1-pP1-pM1
	p0[p0 == 0] = 1e-10
	pP1[pP1 == 0] = 1e-10
	pM1[pM1 == 0] = 1e-10

	P = np.array([p0, pP1, pM1]).flatten()
	H = - (P * np.log2(P))
	H[(P < 2.2204e-16) | (P > 1 - 2.2204e-16)] = 0

	Ht = sum(H)
	return Ht

if __name__ == '__main__':
    path = '/home/fanzexin/XuNet/stego_hill_boss_bow_0.4_AUG/model_params.pt'

    main(path)