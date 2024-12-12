#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:10:55 2024

@author: tonyshara
"""

import torch
import torch.nn as nn


import pandas as pd



# from os import listdir
# import matplotlib.image as mpimg
# from matplotlib.animation import FuncAnimation as FA
# from scipy.signal import butter, filtfilt

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)





class LSTMRegressor(nn.Module):
    def __init__(self, _n_features, _hidden_units):
        super().__init__()
        self.n_features = _n_features
        self.hidden_units = _hidden_units
        self.n_layers = 1
        self.lstm = \
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


