#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:14:14 2024

@author: tonyshara
"""
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import numpy as np

def simple_plot_summary(_df,show=True):
    fig, axes = plt.subplots(8,3, figsize = (15,25), constrained_layout=True, sharex=True)
    for j,c in enumerate(_df.columns[2:-1]):
        axes[j//3][j%3].plot(_df['time'], _df[c])
        axes[j//3][j%3].set_title(c)
    plt.savefig('./figs/plot_summary')
    if show:
        plt.show()
    else:
        plt.close()

def comparison_plot_summary(_samples, _labels,show=True):
    s1,s2,s3 = _samples
    l1,l2,l3 = _labels
    features = s1.columns[4:-1]
    fig, axes = plt.subplots(5,3, figsize = (15,25), constrained_layout=True, sharex=True)
    fig.suptitle('Error Summary', fontsize=30)
    for j,c in enumerate(features):
        axes[j // 3, j % 3].plot(s1['time'], s1[c]-s3[c], c = 'lightblue')
        axes[j // 3, j % 3].set_title(c, fontsize=20)
    plt.savefig('./figs/error_summary')
    if show:
        plt.show()
    else:
        plt.close()
     
    fig, axes = plt.subplots(5,3, figsize = (15,25), constrained_layout=True, sharex=True)
    fig.suptitle('Preprocessing Comparison', fontsize=30)

    for j,c in enumerate(features):
        axes[j // 3, j % 3].plot(s1['time'], s1[c], c = 'lightblue', label = l1)
        axes[j // 3, j % 3].plot(s2['time'], s2[c], c = 'salmon', label = l2)
        axes[j // 3, j % 3].plot(s3['time'], s3[c], c = 'orange', label = l3)
        axes[j // 3, j % 3].set_title(c, fontsize=20)
        axes[j // 3, j % 3].legend(fontsize=12)
    plt.savefig('./figs/comparison_summary')
    if show:
        plt.show()
    else:
        plt.close()   
     
    c = 'Physical Fan Speed'
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title(c)
    ax.plot(s1['time'], s1[c], c = 'lightblue', label = l1)
    ax.plot(s2['time'], s2[c], c = 'salmon'   , label = l2)
    ax.plot(s3['time'], s3[c], c = 'orange'   , label = l3)
    ax.set_title(c)
    plt.savefig(f'./figs/comparison_summary_{c}')
    if show:
        plt.show()
    else:
        plt.close()

    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.set_title(c)
    ax.plot(s1['time'], s1[c]-s3[c], c = 'lightblue')
    plt.savefig(f'./figs/error_summary_{c}')
    if show:
        plt.show()
    else:
        plt.close()


def plot_fft(signal, name='', show=False):
    # Compute the Fourier transform & frequencies
    fft_signal = fft(signal)
    freq = fftfreq(len(signal), 0.001)
    
    # Plot the magnitude spectrum
    plt.plot(freq, np.abs(fft_signal))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Fourier Spectrum " + name)
    plt.savefig(f'./figs/fft_{name}'.replace('/','_'))
    plt.close()
    
    plt.title('Frequency Response')
    plt.magnitude_spectrum(signal)
    plt.savefig(f'./figs/fft_{name}'.replace('/','_'))
    if show:
        plt.show()
    else:
        plt.close()


def plot_lpf_summary(num,den,show=False):
    # Compute the impulse response
    t, y = signal.impulse((num, den))

    # Compute the frequency response
    w, h = signal.freqz(num, den)
    fc = w[np.argmin(np.abs(np.abs(h) - 0.5*np.sqrt(2)))]

    # Plot the frequency response (magnitude and phase)
    fig, ax = plt.subplots(1, constrained_layout=True, sharex=True)
    ax.plot(w, np.abs(h))
    ax.set_title('Low Pass Filter Frequency Response')
    ax.set_xlabel('Frequency (rad/sample)')
    ax.set_ylabel('Magnitude')
    ax.grid(True, which="both", ls="-")
    ax.plot(fc, 0.5*np.sqrt(2), 'ko')
    ax.axvline(fc, color='k')
    ax.set_xscale('log')
    ax.text(0.5, 0.8, f'fc = {fc:.4f} Hz', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', pad=10))
    plt.savefig('./figs/FrequencyResponse')
    if show:
        plt.show()
    else:
        plt.close()
