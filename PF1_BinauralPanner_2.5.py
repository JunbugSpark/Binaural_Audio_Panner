##Code Clean-up 2
##Goal - Change input so that it takes stereo input.

import numpy as np          
import scipy.signal as signal    
import soundfile as sf              
import matplotlib.pyplot as plt   
import os                           
import sounddevice as sd          
from scipy.io import wavfile       
from scipy.signal import resample_poly
import math           

import zipfile              

script_dir = os.path.dirname(os.path.abspath(__file__))

fs, xsig = wavfile.read(os.path.join(script_dir,'Stereo Song 2.wav'))
x = np.array(xsig)

x = x.astype(np.float32)
x = x / np.max(np.abs(x))

x_left = x[:, 0]
x_right = x[:, 1]

elevation = 0
pinna = "H"
azimuth = -150

azimuthL = azimuth - 15
azimuthR = azimuth + 15

tempAzimuthL = azimuthL
tempAzimuthR = azimuthR

if (azimuthL < 0):
  azimuthL = np.abs(azimuthL)
if (azimuthR < 0):
  azimuthR = np.abs(azimuthR)

filenameL = f"elev{elevation}/{pinna}{elevation}e{azimuthL:03d}a.wav"
filenameR = f"elev{elevation}/{pinna}{elevation}e{azimuthR:03d}a.wav"

hfs, hsig_left = wavfile.read(os.path.join(script_dir, filenameL))
hfs, hsig_right = wavfile.read(os.path.join(script_dir, filenameR))

h_sig_left = resample_poly(hsig_left, fs, hfs)
h_sig_right = resample_poly(hsig_right, fs, hfs)

h_left = np.array(h_sig_left)
h_right = np.array(h_sig_right)

h_left = h_left.astype(np.float32)
h_left = h_left / np.max(np.abs(h_left))

h_right = h_right.astype(np.float32)
h_right = h_right / np.max(np.abs(h_right))

h_left_L = h_left[:, 0]
h_left_R = h_left[:, 1]
h_right_L = h_right[:, 0]
h_right_R = h_right[:, 1]

if (tempAzimuthL < 0):
  h_left_L = h_left[:, 1]
  h_left_R = h_left[:, 0]
if (tempAzimuthR < 0):
  h_right_L = h_right[:, 1]
  h_right_R = h_right[:, 0]

def left_convolve(chunk, h_left):

  x_length = len(chunk)
  h_length = len(h_left)
  y_length = x_length + h_length - 1

  y = np.zeros(y_length)

  chunkSize = 1024
  index = 0
  
  for n in range(math.ceil(x_length / chunkSize)):

    leftOne = signal.fftconvolve(chunk, h_left_L, mode='full')
    leftTwo = signal.fftconvolve(chunk, h_right_L, mode='full')
    leftProduct = leftOne + leftTwo

    y[index : index + len(leftProduct)] += leftProduct

    index = index + chunkSize

  return y

def right_convolve(chunk, h_right):
  x_length = len(chunk)
  h_length = len(h_right)
  y_length = x_length + h_length - 1

  y = np.zeros(y_length)

  chunkSize = 1024
  index = 0
  
  for n in range(math.ceil(x_length / chunkSize)):

    rightOne = signal.fftconvolve(chunk, h_right_R, mode='full')
    rightTwo = signal.fftconvolve(chunk, h_left_R, mode='full')
    rightProduct = rightOne + rightTwo

    y[index : index + len(rightProduct)] += rightProduct

    index = index + chunkSize

  return y





tail_left = np.zeros(len(h_left) - 1)
tail_right = np.zeros(len(h_right) - 1)

position = 0

##Callback Structure
def callbackmachine(outdata, frames, time, status):
  global position
  global tail_left
  global tail_right

  chunkleft = x_left[position : position + frames]
  chunkright = x_right[position : position + frames]

  left_out = left_convolve(chunkleft, h_left)
  right_out = right_convolve(chunkright, h_right)

  outdata[:, 0] = left_out[: frames]
  outdata[: len(tail_left), 0] += tail_left
  outdata[:, 1] = right_out[: frames]
  outdata[: len(tail_right), 1] += tail_right

  tail_left = left_out[frames : len(left_out)]
  tail_right = right_out[frames : len(right_out)]  
  
  position += frames

with sd.OutputStream(samplerate = fs, blocksize = 512, channels = 2, callback = callbackmachine):
  duration = int(len(x) / fs * 1000)
  sd.sleep(duration)
