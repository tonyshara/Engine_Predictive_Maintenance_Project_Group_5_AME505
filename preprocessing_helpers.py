#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:08:49 2024

@author: tonyshara
"""
import torch
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def minMax(_df):
    minmax_dict = {}
    for c in _df.columns[4:-1]:
        minmax_dict[c+'_min'] = _df[c].min()
        minmax_dict[c+'_max'] =  _df[c].max()
    
    ret = _df.copy()
    for c in _df.columns[4:-1]:
        ret[c] = (ret[c] - minmax_dict[c+'_min']) / (minmax_dict[c+'_max'] - minmax_dict[c+'_min'])
        
    return ret

#Smoothing Function: Exponentially Weighted Averages
def smooth(s, b = 0.98):
    v = np.zeros(len(s)+1) #v_0 is already 0.
    bc = np.zeros(len(s)+1)
    for i in range(1, len(v)): #v_t = 0.95
        v[i] = (b * v[i-1] + (1-b) * s[i-1])
        bc[i] = 1 - b**i
    sm = v[1:] / bc[1:]
    return sm  

# LPF Filtering Function: Filter out high frequency noise
def butter_lowpass(cutoff=5, fs=1000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def apply_filter(data, b, a):
    y = filtfilt(b, a, data)
    return y    

#Smoothing each time series for each engine in both training and test sets
def get_smoothed_data(_df_train, _df_test):
    df_train_smoothed = pd.DataFrame()
    df_train_smoothed['Engine Unit'] = _df_train['Engine Unit']
    df_train_smoothed['time'] = _df_train['time']
    df_train_smoothed['os1'] = _df_train['os1']
    df_train_smoothed['os2'] = _df_train['os2']
    df_test_smoothed = pd.DataFrame()
    df_test_smoothed['Engine Unit'] = _df_test['Engine Unit']
    df_test_smoothed['time'] = _df_test['time']
    df_test_smoothed['os1'] = _df_test['os1']
    df_test_smoothed['os2'] = _df_test['os2']
    for c in _df_train.columns[4:-1]:
      sm_list = []
      for n in np.arange(1,101):
        s = np.array(_df_train[_df_train['Engine Unit'] == n][c].copy())
        sm = list(smooth(s, 0.98))
        sm_list += sm
      df_train_smoothed[c] = sm_list
    for c in _df_test.columns[4:-1]:
      sm_list = []
      for n in np.arange(1,101):
        s = np.array(_df_test[_df_test['Engine Unit'] == n][c].copy())
        sm = list(smooth(s, 0.98))
        sm_list += sm
      df_test_smoothed[c] = sm_list

    df_train_smoothed['rul'] = _df_train['rul']
    df_test_smoothed['rul'] = _df_test['rul']
    return df_train_smoothed, df_test_smoothed

#Filtering each time series for each engine in both training and test sets
def get_LPF_filtered_data(_df_train, _df_test, cutoff_low=5, fs=1000, order=5):
    df_train_LPF = pd.DataFrame()
    df_train_LPF['Engine Unit'] = _df_train['Engine Unit']
    df_train_LPF['time'] = _df_train['time']
    df_train_LPF['os1'] = _df_train['os1']
    df_train_LPF['os2'] = _df_train['os2']
    df_test_LPF = pd.DataFrame()
    df_test_LPF['Engine Unit'] = _df_test['Engine Unit']
    df_test_LPF['time'] = _df_test['time']
    df_test_LPF['os1'] = _df_test['os1']
    df_test_LPF['os2'] = _df_test['os2']
    
    b, a = butter_lowpass(cutoff_low, fs, order)
    for c in _df_train.columns[4:-1]:
      f_list = []
      for n in np.arange(1,101):
        s = np.array(_df_train[_df_train['Engine Unit'] == n][c].copy())
        f = list(apply_filter(s, b, a))
        f_list += f
      df_train_LPF[c] = f_list
    for c in _df_test.columns[4:-1]:
      f_list = []
      for n in np.arange(1,101):
        s = np.array(_df_test[_df_test['Engine Unit'] == n][c].copy())
        f = list(apply_filter(s, b, a))
        f_list += f
      df_test_LPF[c] = f_list

    df_train_LPF['rul'] = _df_train['rul']
    df_test_LPF['rul'] = _df_test['rul']
    return df_train_LPF, df_test_LPF







def load_data(_train_data, _val_data, _df_train, _window):
    torch.manual_seed(5)
    train_indices = list(_train_data[(_train_data['rul'] >= (_window - 1)) & (_train_data['time'] > 10)].index)
    val_indices = list(_val_data[(_val_data['rul'] >= (_window - 1)) & (_val_data['time'] > 10)].index)
    train = data(train_indices, _df_train)
    val = data(val_indices, _df_train)
    t = DataLoader(train, batch_size = 64, shuffle = True)
    v = DataLoader(val, batch_size = len(val_indices), shuffle = True)
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
    
class LSTMRegressor(nn.Module):
    def __init__(self, _n_features, _hidden_units):
        super().__init__()
        self.n_features = _n_features
        self.hidden_units = _hidden_units
        self.n_layers = 1
        self.lstm =\
            nn.LSTM(input_size=self.n_features,
                    hidden_size=self.hidden_units,
                    batch_first=True,
                    num_layers=self.n_layers)
        self.linear1 = nn.Linear(in_features=self.hidden_units, out_features=12)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=12, out_features=12)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=12, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_units, device=x.device).requires_grad_()
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_units, device=x.device).requires_grad_()
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear1(hn[0])
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out).flatten()
        return out

learning_rate = 0.001
n_hidden_units = 12
def get_model_training_helpers(_n_features, _device, _learning_rate=learning_rate, _n_hidden_units=n_hidden_units):
    torch.manual_seed(15)

    model = LSTMRegressor(_n_features, _n_features).to(_device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_learning_rate)

    ks = [key for key in model.state_dict().keys() if 'linear' in key and '.weight' in key]

    for k in ks:
        nn.init.kaiming_uniform_(model.state_dict()[k])

    bs = [key for key in model.state_dict().keys() if 'linear' in key and '.bias' in key]

    for b in bs:
        nn.init.constant_(model.state_dict()[b], 0)
    return model, loss_fn, optimizer
    

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

