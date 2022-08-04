# -*- coding: utf-8 -*-
import pickle as pkl
import pandas as pd
import numpy as np
import math
import os
import datetime as dt
import numpy.linalg as la
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tgcn_res_ass_att import ass_TGCN
from input_data import preprocess_data_ass_js, load_js_data
from my.synthetic_dataset import Dataset_ass
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time
CUDA_VISIBLE_DEVICES = 0
time_start = time.time()
###### Settings ######
model_name = 'tgcn'
data_name = 'js'
train_rate = 0.8
seq_len = 12
pre_len =3
batch_size = 50
lr = 0.001
training_epoch = 60
hidden_dim = 64
station_num = 31
pever = 5
pat = 20
save = 'F'   #True
loss_type = 'mse'  # dilate  #mse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
###### load data ######
if data_name == 'js':
    data, adj, dir = load_js_data('js')


def prep_clf(obs, pre, threshold):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # 根据阈值分类为 0, 1
    obs = np.where(obs > threshold, 1, 0)
    pre = np.where(pre > threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives


def TS(obs, pre, threshold):
    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses)
    	  alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return hits / (hits + falsealarms + misses)

def train_model(net, loss_type, learning_rate, epochs, gamma=0.1,
                print_every=1, eval_every=50, verbose=1, Lambda=1, alpha=0.5):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler_enc = ReduceLROnPlateau(optimizer, mode='min', patience=pat, factor=0.5, verbose=True)
    criterion = torch.nn.MSELoss()
    criterion1 = torch.nn.L1Loss()
    min_loss = 1000
    loss_a, r_loss = [], []
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, target = data
            inputs = inputs.clone().detach().cuda().type(dtype=torch.float32)
            target = target.clone().detach().cuda().type(dtype=torch.float32)
            # forward + backward + optimize
            outputs = net(inputs)
            if (loss_type == 'mse'):
                loss = criterion(target, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (verbose):
            if (epoch % print_every == 0):
                print('epoch ', epoch)
                real_loss = eval_model(net, testloader, gamma, loss_type=loss_type, verbose=1)
                scheduler_enc.step(real_loss)
                r_loss.append(real_loss)
                if optimizer.param_groups[0]['lr'] <= 0.00001:  # 0.00001
                    break
                if real_loss < min_loss:
                    min_loss = real_loss
                    print("save model")
                    torch.save(net.state_dict(), 'model.pth')
    r_loss= pd.DataFrame(r_loss)
    r_loss.to_csv('loss.csv')

def eval_model(net, loader, gamma, loss_type, verbose=1):
    losses_mse = []
    for i, data in enumerate(loader, 0):
        inputs, target = data
        inputs = inputs.clone().detach().cuda().type(dtype=torch.float32)
        outputs = net(inputs)  # *max_value
        if (loss_type == 'mse'):
            outputs1 = outputs.detach().cpu().numpy().squeeze()
            target1 = target.cpu().numpy().squeeze()
            losses_mse.append(mean_squared_error(outputs1, target1))
    if (loss_type == 'mse'):
        print(' Eval mse= ', np.array(losses_mse).mean())
    # if (loss_type == 'dilate'):
    #     print(' Eval mse= ', np.array(losses_mse).mean(), ' dtw= ', np.array(losses_dtw).mean(), ' tdi= ',
    #           np.array(losses_tdi).mean())
    return np.array(losses_mse).mean()


time_len = int(data.shape[0])
num_nodes = data.shape[1]
data1 = np.mat(data, dtype=np.float32)

#### normalization
max_value = np.max(data)
data1 = data / max_value

if data_name == 'js':   #js
    X_train_input, X_train_target, X_test_input, X_test_target = preprocess_data_ass_js(data1, time_len, train_rate, seq_len,
                                                                                 pre_len)
test_input = np.array(X_test_input)
train_input = np.array(X_train_input)
train_target = np.array(X_train_target)
test_target = np.array(X_test_target)

def test_model(testloader):
    input, target, preds = [], [], []
    nets = net_tgcn  # net_gru_mse,
    state_dict = torch.load('model.pth')
    nets.load_state_dict(state_dict)
    for i, gen_test in enumerate(testloader, 0):
        test_inputs, test_targets = gen_test
        test_inputs = test_inputs.clone().detach().cuda().type(dtype=torch.float32)
        outputs = nets(test_inputs).to(device)
        pred = outputs  # *max_value
        target.append(test_targets.detach().cpu().numpy().squeeze())  # [ind,:,:][:, ind, :]
        preds.append(pred.detach().cpu().numpy().squeeze())  # [:, ind, :]
    return np.array(preds), np.array(target)

def draw_all(preds1, target1):  # ,max_value
    print('ALL_TS:')
    print('TS0.1:', TS(target1, preds1, 0.1))
    print('TS1.5:', TS(target1, preds1, 1.5))
    print('TS3.0:', TS(target1, preds1, 3))
    print('TS7.0:', TS(target1, preds1, 7))
    print('TS10.0:', TS(target1, preds1, 10))
    print('TS14.9:', TS(target1, preds1, 14.9))
    print('TS20.0:', TS(target1, preds1, 20.0))
    print('TS50.0:', TS(target1, preds1, 50.0))
    print('mse_all:', mean_squared_error(target1,preds1 ),'mae_all:',mean_absolute_error(target1,preds1))

dataset_train = Dataset_ass(X_train_input, X_train_target)
dataset_test = Dataset_ass(X_test_input, X_test_target)
trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
testloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
net_tgcn = ass_TGCN(adj=adj, hidden_dim=hidden_dim,seq_len=seq_len, pre_len=pre_len, wind=dir).to(device)  # ,K=2
train_model(net_tgcn, loss_type=loss_type, learning_rate=lr, epochs=training_epoch, gamma=0.01,
            print_every=pever, eval_every=pever, verbose=1)
preds1, target1 = test_model(testloader)
preds1 = preds1 * max_value[np.newaxis, :]
target1 = target1 * max_value[np.newaxis, :]
preds1[preds1 < 0.1] = 0
if pre_len > 1:
    preds1 = preds1.sum(axis=1)
    target1 = target1.sum(axis=1)
if save == 'True':
    pred1 = pd.DataFrame(preds1)
    pred1.to_csv(str(hidden_dim)+str(pre_len)+'h.csv')

