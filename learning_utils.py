#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:42:34 2024

@author: tonyshara
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import torch.optim as optim
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

class data(Dataset):

    def __init__(self, _list_indices, _df_train):

        self.indices = _list_indices
        self.df_train = _df_train

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        ind = self.indices[idx]
        X_ = self.df_train.iloc[ind : ind + 20, :].drop(['time','Engine Unit','rul'], axis = 1).copy().to_numpy()
        y_ = self.df_train.iloc[ind + 19]['rul']

        return X_, y_
def get_indices(_train_data, _val_data, _window):
    t = list(_train_data[(_train_data['rul'] >= (_window - 1)) & (_train_data['time'] > 10)].index)
    v = list(_val_data[(_val_data['rul'] >= (_window - 1)) & (_val_data['time'] > 10)].index)
    return t,v


def get_dataLoader(_train_data, _val_data, _df_train, _window):
    torch.manual_seed(5)
    train_indices = list(_train_data[(_train_data['rul'] >= (_window - 1)) & (_train_data['time'] > 10)].index)
    val_indices = list(_val_data[(_val_data['rul'] >= (_window - 1)) & (_val_data['time'] > 10)].index)
    train = data(train_indices, _df_train)
    val = data(val_indices, _df_train)
    t = DataLoader(train, batch_size = 64, shuffle = True)
    v = DataLoader(val, batch_size = len(val_indices), shuffle = True)
    return t,v

class data_test(Dataset):

    def __init__(self, _units, _df_test):

        self.units = _units
        self.df_test = _df_test

    def __len__(self):

        return len(self.units)

    def __getitem__(self, idx):

        n = self.units[idx]
        U = self.df_test[self.df_test['Engine Unit'] == n].copy()
        X_ = U.reset_index().iloc[-20:,:].drop(['index','Engine Unit','time','rul'], axis = 1).copy().to_numpy()
        y_ = U['rul'].min()
        return X_, y_
    
def validation(_model, _loss_fn, _valloader, _device):
    _model.eval()
    X, y = next(iter(_valloader))
    X, y = X.to(_device).to(torch.float32), y.to(_device).to(torch.float32)

    with torch.no_grad():
        y_pred = _model(X).to(_device).to(torch.float32)
        val_loss = _loss_fn(y_pred, y).item()

    return val_loss

def test_function(_model, _loss_fn, _testloader, _device):
    loss_L1 = nn.L1Loss()

    _model.eval()
    X, y = next(iter(_testloader))
    X, y = X.to(_device).to(torch.float32), y.to(_device).to(torch.float32)

    with torch.no_grad():
        y_pred = _model(X).to(_device).to(torch.float32)
        test_loss_MSE = _loss_fn(y_pred, y).item()
        test_loss_L1 = loss_L1(y_pred, y).item()

    return test_loss_MSE, test_loss_L1,  y_pred, y

def train_model(_model, _trainloader, _valloader, _device, _optimizer, _loss_fn, epochs=100):
    T = []
    V = []

    for i in tqdm(range(epochs)):
        _model.train()
        L = 0

        for batch, (X,y) in enumerate(_trainloader):

            X, y = X.to(_device).to(torch.float32), y.to(_device).to(torch.float32)

            y_pred = _model(X)
            loss = _loss_fn(y_pred, y)
            L += loss.item()
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

        val_loss = validation(_model, _loss_fn, _valloader, _device)

        T.append(L/len(_trainloader))
        V.append(val_loss)

        if (i+1) % 10 == 0:
            sleep(0.5)
            print(f'epoch:{i+1}, avg_train_loss:{L/len(_trainloader)}, val_loss:{val_loss}')
            _model.train()
    return T, V

def err_analysis(_model, _loss_fn, _testloader, _device, _name='',show=True):
    mse, l1, y_pred, y = test_function(_model, _loss_fn, _testloader, _device)
    print(f'Test MSE:{round(mse,2)}, L1:{round(l1,2)}')
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(np.arange(1,101), y_pred.cpu().numpy(), label = 'predictions', c = 'salmon')
    ax.plot(np.arange(1,101), y.cpu().numpy(), label = 'true values', c = 'lightseagreen')
    ax.set_xlabel('Test Engine Units', fontsize = 16)
    ax.set_ylabel('RUL', fontsize = 16)
    ax.grid(True)
    ax.legend()
    plt.savefig(f'./figs/err_fig_{_name}')
    plt.show()

    # Assume predictions and true_rul are numpy arrays
    true_rul = y.cpu().numpy()
    predictions = y_pred.cpu().numpy()

    errors = true_rul - predictions
    mse = mean_squared_error(true_rul, predictions)
    std_dev = np.std(errors)
    n = len(true_rul)

    # Standard error and confidence interval
    se = std_dev / np.sqrt(n)
    z = 1.96  # For 95% confidence
    confidence_interval = z * se

    print(f"Mean Absolute Error: {np.mean(np.abs(errors))}")
    print(f"95% Confidence Interval: ±{confidence_interval}")

    # Visualization (e.g., using matplotlib)
    plt.plot(predictions, label="Predicted RUL")
    plt.xlabel('Test Engine Units', fontsize = 16)
    plt.ylabel('RUL', fontsize = 16)
    plt.fill_between(range(len(predictions)),
                     predictions - confidence_interval,
                     predictions + confidence_interval,
                     color='gray', alpha=0.2, label="95% CI")
    plt.plot(true_rul, label="True RUL")
    plt.legend()
    if show:
        plt.show()
    else:
        plt.close()

    return mse, std_dev, se, confidence_interval
