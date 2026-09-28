##Code Clean-up
##Goal - Change the 180 degree azimuth so that it has complete 360 degree coverage.

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

fs, xsig = wavfile.read(os.path.join(script_dir,'Sample Song.wav'))
x = np.array(xsig)

x = x / np.max(np.abs(x))

elevation = 10
pinna = "H"
azimuth = -60
##Temporary variable to store a positive / negative azimuth
tempAzimuth = azimuth

if (azimuth < 0):
  azimuth = np.abs(azimuth)

filename = f"elev{elevation}/{pinna}{elevation}e0{azimuth}a.wav"

hfs, hsig = wavfile.read(os.path.join(script_dir, filename))

h_sig = resample_poly(hsig, fs, hfs)

h = np.array(h_sig)

h = h.astype(np.float32)
h = h / np.max(np.abs(h))

h_left = h[:, 0]
h_right = h[:, 1]

##Calling the temporary azimuth variable allows a negative azimuth to still be effective.
if (tempAzimuth < 0):
  h_left = h[:, 1]
  h_right = h[:, 0]

def left_convolve(x, h):

  x_length = len(x)
  h_length = len(h)
  y_length = x_length + h_length - 1

  y = np.zeros(y_length)

  chunkSize = 1024
  index = 0
  
  for n in range(math.ceil(x_length / chunkSize)):

    chunk = x[index : index + chunkSize]

    leftProduct = signal.fftconvolve(chunk, h_left, mode='full')

    y[index : index + len(leftProduct)] += leftProduct

    index = index + chunkSize

  return y

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





tail_left = np.zeros(len(h) - 1)
tail_right = np.zeros(len(h) - 1)

position = 0

##Callback Structure
def callbackmachine(outdata, frames, time, status):
  global position
  global tail_left
  global tail_right

  chunk = x[position : position + frames]

  left_out = left_convolve(chunk, h)
  right_out = right_convolve(chunk, h)

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
