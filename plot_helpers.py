#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:14:14 2024

@author: tonyshara
"""
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np


def simple_plot_summary(_df):
    fig, axes = plt.subplots(8,3, figsize = (15,25), constrained_layout=True, sharex=True)
    for j,c in enumerate(_df.columns[2:-1]):
        axes[j//3][j%3].plot(_df['time'], _df[c])
        axes[j//3][j%3].set_title(c)
    plt.savefig('./figs/plot_summary')
    plt.show()

def comparison_plot_summary(_samples, _labels):
    s1,s2,s3 = _samples
    l1,l2,l3 = _labels
    features = s1.columns[4:-1]
    fig, axes = plt.subplots(6,3, figsize = (15,25))
    fig.tight_layout()
    for j,c in enumerate(features):
        axes[j // 3, j % 3].plot(s1['time'], s1[c], c = 'lightblue', label = 'original')
        axes[j // 3, j % 3].plot(s2['time'], s2[c], c = 'salmon', label = 'smoothed')
        axes[j // 3, j % 3].plot(s3['time'], s3[c], c = 'orange', label = 'LPF')
        axes[j // 3, j % 3].set_title(c)
        axes[j // 3, j % 3].legend()
    plt.savefig('./figs/comparison_summary')
    plt.show()
    
def plot_fft(signal):
  # Compute the Fourier transform
  fft_signal = fft(signal)

  # Compute the frequencies
  freq = fftfreq(len(signal), 0.001)

  # Plot the magnitude spectrum
  plt.plot(freq, np.abs(fft_signal))
  plt.xlabel("Frequency (Hz)")
  plt.ylabel("Magnitude")
  plt.title("Fourier Spectrum")
  plt.savefig('./figs/fft')
  plt.show()