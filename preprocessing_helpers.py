#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:08:49 2024

@author: tonyshara
"""

# import torch.optim as optim

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from plot_helpers import plot_lpf_summary, plot_fft

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
    
    for c in _df_train.columns[4:-1]:
        s4  = np.array(_df_train[_df_train['Engine Unit'] == 4][c].copy())
        s83 = np.array(_df_train[_df_train['Engine Unit'] == 83][c].copy())
        plot_fft(s4,c+'_4')
        plot_fft(s83,c+'_83')
    

    b, a = butter_lowpass(cutoff_low, fs, order)
    plot_lpf_summary(b, a)
    
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












