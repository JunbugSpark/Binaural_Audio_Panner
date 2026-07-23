##Overlap-Add Convolution
##Goal is to speed up the 1.1 convolution process

import numpy as np                  ##Core DSP math library
import scipy.signal as signal       ##Contains pre-built DSP functions
import soundfile as sf              ##Reads and writes .wav files
import matplotlib.pyplot as plt     ##Allows you to plot and visualize
import os                           ##File path handling
import sounddevice as sd            ##Crucial for real time audio
from scipy.io import wavfile        ##For reading .wav files
from scipy.signal import resample_poly  ##For changing HRTF sample rate to 48kHz
import math                         ##For rounding up during the for loop

import zipfile                      ##For unzipping HRTF dataset

script_dir = os.path.dirname(os.path.abspath(__file__))

##Read sample audio file
fs, xsig = wavfile.read(os.path.join(script_dir,'Sample Song.wav'))
x = np.array(xsig)

##Assign HRTF parameters
elevation = 80
pinna = "H"
azimuth = 60

##Assign HRTF filename based on parameters
filename = f"elev{elevation}/{pinna}{elevation}e0{azimuth}a.wav"

##Read the HRTF file
hfs, hsig = wavfile.read(os.path.join(script_dir, filename))

##Resample the HRTF to match audio file sample rate
h_sig = resample_poly(hsig, fs, hfs)

##Divide the HRTF into left and right channels
h = np.array(h_sig)

h_left = h[:, 0]
h_right = h[:, 1]

##Stereo Left Convolve
def left_convolve(x, h):
  ##Initialize array lengths
  x_length = len(x)
  h_length = len(h)
  y_length = x_length + h_length - 1

  ##Array that output will go into
  y = np.zeros(y_length)

  ##Chunk size inputs, decided by user depending on needs
  chunkSize = 1024
  index = 0
  
  for n in range(math.ceil(x_length / chunkSize)):

    chunk = x[index : index + chunkSize]

    leftProduct = signal.fftconvolve(chunk, h_left, mode='full')

    y[index : index + len(leftProduct)] += leftProduct

    index = index + chunkSize

  return y

##Stereo-right convolve
def right_convolve(x, h):
  x_length = len(x)
  h_length = len(h)
  y_length = x_length + h_length - 1

  y = np.zeros(y_length)

  chunkSize = 1024
  index = 0
  
  for n in range(math.ceil(x_length / chunkSize)):

    chunk = x[index : index + chunkSize]

    leftProduct = signal.fftconvolve(chunk, h_right, mode='full')

    y[index : index + len(leftProduct)] += leftProduct

    index = index + chunkSize

  return y

y_left = left_convolve(x, h)
y_right = right_convolve(x, h)

output = np.zeros((len(y_left), 2))

output[:, 0] = y_left
output[:, 1] = y_right

output = output / np.max(np.abs(output))

print("Shape:", output.shape, "Dtype:", output.dtype)

sd.play(output, fs)
sd.wait()
