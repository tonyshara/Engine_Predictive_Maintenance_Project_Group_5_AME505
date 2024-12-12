#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:13:41 2024

@author: tonyshara
"""


import pandas as pd
import numpy as np
# from scipy.signal import butter, filtfilt
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from time import sleep
from os import listdir
# import matplotlib.image as mpimg
# from matplotlib.animation import FuncAnimation as FA

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)


# from gru_helpers import *
from plot_helpers import comparison_plot_summary
from preprocessing_helpers import minMax, get_smoothed_data, get_LPF_filtered_data

col_names = [
  'Engine Unit',
  'time',
  'os1','os2','os3',
  'Fan Inlet Temp',                           #s1
  'LPC Outlet Temp',                          #s2
  'HPC Outlet Temp',                          #s3
  'LPT Outlet Temp',                          #s4
  'Fan Inlet Pressure',                       #s5
  'Bypass-Duct Pressure',                     #s6
  'HPC Outlet Pressure',                      #s7
  'Physical Fan Speed',                       #s8
  'Physical Core Speed',                      #s9
  'Engine Pressure Ratio (P50/P2)',           #s10
  'HPC Outlet Static Pressure',               #s11
  'Ratio of Fuel Flow to Ps30 (pps/psia)',    #s12
  'Corrected Fan Speed',                      #s13
  'Corrected Core Speed',                     #s14
  'Bypass Ratio',                             #s15
  'Burner Fuel-Air Ratio',                    #s16
  'Bleed Enthalpy',                           #s17
  'Required Fan Speed',                       #s18
  'Required Fan Conversion Speed',            #s19
  'High-Pressure Turbines Cool Air Flow',     #s20
  'Low-Pressure Turbines Cool Air Flow'       #s21
]

drop_cols1 = [
  'os3',
  'Fan Inlet Temp',                           #s1
  'Fan Inlet Pressure',                       #s5
  'Bypass-Duct Pressure',                     #s6
  'Engine Pressure Ratio (P50/P2)',           #s10
  'Burner Fuel-Air Ratio',                    #s16
  'Required Fan Speed',                       #s18
  'Required Fan Conversion Speed',            #s19
]

def get_rul_test_train(_df_test, _rul_test, _df_train):
  rul_list_train = []
  for n in np.arange(1,101):
      time_list = np.array(_df_train[_df_train['Engine Unit'] == n]['time'])
      length = len(time_list)
      rul = list(length - time_list)
      rul_list_train += rul
  rul_list_test = []

  for n in np.arange(1,101):
      time_list = np.array(_df_test[_df_test['Engine Unit'] == n]['time'])
      length = len(time_list)
      rul_val = _rul_test.iloc[n-1].item()
      rul = list(length - time_list + rul_val)
      rul_list_test += rul
  return rul_list_test, rul_list_train



SAMPLE = 4
if __name__ == '__main__':

    # LOAD DATA
    folder_path = './CMAPS/'
    listdir(folder_path)
    df_train_fd001 = pd.read_csv(folder_path + 'train_FD001.txt', header = None, sep = ' ')
    df_train_fd002 = pd.read_csv(folder_path + 'train_FD002.txt', header = None, sep = ' ')
    df_train_fd003 = pd.read_csv(folder_path + 'train_FD003.txt', header = None, sep = ' ')
    df_train_fd004 = pd.read_csv(folder_path + 'train_FD004.txt', header = None, sep = ' ')
    df_test_fd001 = pd.read_csv(folder_path + 'test_FD001.txt', header = None, sep = ' ')
    df_test_fd002 = pd.read_csv(folder_path + 'test_FD002.txt', header = None, sep = ' ')
    df_test_fd003 = pd.read_csv(folder_path + 'test_FD003.txt', header = None, sep = ' ')
    df_test_fd004 = pd.read_csv(folder_path + 'test_FD004.txt', header = None, sep = ' ')
    rul_test_fd001 = pd.read_csv(folder_path + 'RUL_FD001.txt', header = None)
    rul_test_fd002 = pd.read_csv(folder_path + 'RUL_FD002.txt', header = None)
    rul_test_fd003 = pd.read_csv(folder_path + 'RUL_FD003.txt', header = None)
    rul_test_fd004 = pd.read_csv(folder_path + 'RUL_FD004.txt', header = None)
    df_train_fd001 = df_train_fd001.iloc[:,:-2].copy()
    df_train_fd002 = df_train_fd002.iloc[:,:-2].copy()
    df_train_fd003 = df_train_fd003.iloc[:,:-2].copy()
    df_train_fd004 = df_train_fd004.iloc[:,:-2].copy()
    df_train_fd001.columns = col_names
    df_train_fd002.columns = col_names
    df_train_fd003.columns = col_names
    df_train_fd004.columns = col_names
    df_test_fd001 = df_test_fd001.iloc[:,:-2].copy()
    df_test_fd002 = df_test_fd002.iloc[:,:-2].copy()
    df_test_fd003 = df_test_fd003.iloc[:,:-2].copy()
    df_test_fd004 = df_test_fd004.iloc[:,:-2].copy()
    df_test_fd001.columns = col_names
    df_test_fd002.columns = col_names
    df_test_fd003.columns = col_names
    df_test_fd004.columns = col_names



    ##############################
    #   Dataset & Preprocessing
    ##############################

    # Get data frames & rul lists
    df_test  = df_test_fd001
    df_train = df_train_fd001
    rul_test = rul_test_fd001
    rul_list_test, rul_list_train = get_rul_test_train(df_test, rul_test, df_train)
    df_test['rul'], df_train['rul'] = rul_list_test, rul_list_train

    # Chose sample engine
    # sample_df = df_train[df_train['Engine Unit'] == SAMPLE].copy()

    # 1) Chose features
    df_train = df_train.drop(drop_cols1, axis = 1)
    df_test = df_test.drop(drop_cols1, axis = 1)

    # 2) MinMax Scaling
    df_train = minMax(df_train)
    df_test  = minMax(df_test)

    # 3) Smoothing: Exponentially Weighted Average
    df_train_smoothed, df_test_smoothed = get_smoothed_data(df_train, df_test)

    # 4) Low Pass Filter
    df_train_LPF, df_test_LPF = get_LPF_filtered_data(df_train, df_test, cutoff_low=12, fs=1000, order=5)

    # 5) Preprocessing
    sample = 10
    sample_df               = df_train[df_train['Engine Unit'] == sample].copy()
    smoothed_sample_df      = df_train_smoothed[df_train_smoothed['Engine Unit'] == sample].copy()
    LPF_sample_df           = df_train_LPF[df_train_LPF['Engine Unit'] == sample].copy()

    # Sample Data
    samples = [sample_df,smoothed_sample_df,LPF_sample_df]
    labels = ['original','smoothed','LPF']
    comparison_plot_summary(samples, labels)
    
    ######################
    # LSTM
    ######################
    RUN_LSTM = False
    if RUN_LSTM:
        import torch
        from torch.utils.data import DataLoader
        from learning_utils import get_dataLoader, data_test, train_model, err_analysis
        from lstm_helpers import  get_model_training_helpers
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
        # 2) Model Parameters
        n_features = len(df_train.columns[2:-1])
        window = 20
        print(f'number of features: {n_features}, window size: {window}')
        np.random.seed(5)
        units = np.arange(1,101)
        train_units = list(np.random.choice(units, 80, replace = False))
        val_units = list(set(units) - set(train_units))
        print(val_units)
    
    
        # 3) Original Model
        train_data = df_train[df_train['Engine Unit'].isin(train_units)].copy()
        val_data = df_train[df_train['Engine Unit'].isin(val_units)].copy()
        trainloader,valloader = get_dataLoader(train_data, val_data, df_train, window)
        test = data_test(units, df_test)
        testloader = DataLoader(test, batch_size = 100)
        model, loss_fn, optimizer = get_model_training_helpers(n_features, device)
        train_model(model, trainloader, valloader, device, optimizer, loss_fn, epochs=100)
    
        # Save the model to the specified file
        torch.save(model.state_dict(), 'model.pth')
        # Load Models
        model.load_state_dict(torch.load('model.pth'))
    
        err_analysis(model, loss_fn, testloader, device, _name='original')
    
        # 4) Smoothed Model
        train_data_sm = df_train_smoothed[df_train_smoothed['Engine Unit'].isin(train_units)].copy()
        val_data_sm = df_train_smoothed[df_train_smoothed['Engine Unit'].isin(val_units)].copy()
        trainloader_sm,valloader_sm = get_dataLoader(train_data_sm, val_data_sm, df_train_smoothed, window)
        test_sm = data_test(units, df_test)
        testloader_sm = DataLoader(test_sm, batch_size = 100)
        model_sm, loss_fn_sm, optimizer_sm = get_model_training_helpers(n_features, device)
        train_model(model_sm, trainloader_sm, valloader_sm, device, optimizer_sm, loss_fn_sm, epochs=100)
    
        # Save the model to the specified file
        torch.save(model_sm.state_dict(), 'model_smoothed.pth')
        # Load Models
        model_sm.load_state_dict(torch.load('model_smoothed.pth'))
    
        err_analysis(model_sm, loss_fn_sm, testloader_sm, device, _name='smoothed')
    
        # 5) LPF Model
        train_data_LPF = df_train_LPF[df_train_LPF['Engine Unit'].isin(train_units)].copy()
        val_data_LPF = df_train_LPF[df_train_LPF['Engine Unit'].isin(val_units)].copy()
        trainloader_LPF,valloader_LPF = get_dataLoader(train_data_LPF, val_data_LPF, df_train_LPF, window)
        test_LPF = data_test(units, df_test)
        testloader_LPF = DataLoader(test_LPF, batch_size = 100)
        model_LPF, loss_fn_LPF, optimizer_LPF = get_model_training_helpers(n_features, device)
        train_model(model_LPF, trainloader_LPF, valloader_LPF, device, optimizer_LPF, loss_fn_LPF, epochs=100)
    
        # Save the model to the specified file
        torch.save(model_LPF.state_dict(), 'model_LPF.pth')
        # Load Models
        model_LPF.load_state_dict(torch.load('model_LPF.pth'))
    
        err_analysis(model_LPF, loss_fn_LPF, testloader_LPF, device, _name='LPF')
