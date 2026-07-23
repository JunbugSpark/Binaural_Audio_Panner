##Regular Convolution

import numpy as np                  ##Core DSP math library
import scipy.signal as signal       ##Contains pre-built DSP functions
import soundfile as sf              ##Reads and writes .wav files
import matplotlib.pyplot as plt     ##Allows you to plot and visualize
import os                           ##File path handling
import sounddevice as sd            ##Crucial for real time audio
from scipy.io import wavfile        ##For reading .wav files
from scipy.signal import resample_poly  ##For changing HRTF sample rate to 48kHz

import zipfile                      ##For unzipping HRTF dataset

zipfile.ZipFile('/Users/junbug/Documents/VS Code/Python/HRTF.zip').extractall('/Users/junbug/Documents/VS Code/Python')

fs, xsig = wavfile.read('/Users/junbug/Documents/VS Code/Python/XO Tour Llif3.wav')
x = np.array(xsig)

elevation = 80
pinna = "H"
azimuth = 60

filename = f"elev{elevation}/{pinna}{elevation}e0{azimuth}a.wav"

hfs, hsig = wavfile.read('/Users/junbug/Documents/VS Code/Python/' + filename)

h_sig = resample_poly(hsig, fs, hfs)

h = np.array(h_sig)

h_left = h[:, 0]

def left_convolve(x, h):
  x_length = len(x)
  h_length = len(h)
  y_length = x_length + h_length - 1

  y = np.zeros(y_length)

  for n in range(y_length):
    for k in range(h_length):
      if 0 <= n - k < x_length:
        y[n] += h_left[k] * x[n - k]

  return y

y_left = left_convolve(x, h)

h_right = h[:, 1]

def right_convolve(x, h):
  x_length = len(x)
  h_length = len(h)
  y_length = x_length + h_length - 1

  y = np.zeros(y_length)

  for n in range(y_length):
    for k in range(h_length):
      if 0 <= n - k < x_length:
        y[n] += h_right[k] * x[n - k]

  return y

y_right = right_convolve(x, h)

output = np.zeros((len(y_left), 2))

output[:, 0] = y_left
output[:, 1] = y_right

output = output / np.max(np.abs(output))

sd.play(output, fs)
sd.wait()