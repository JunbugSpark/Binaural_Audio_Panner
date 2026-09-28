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

##Locates the song and makes it an audio file (xsig)
fs, xsig = wavfile.read(os.path.join(script_dir,'Stereo Song 2.wav'))
##Turns signal into an array
x = np.array(xsig)

##Normalize
x = x.astype(np.float32)
x = x / np.max(np.abs(x))

##Stereo rray split into left and right channels
x_left = x[:, 0]
x_right = x[:, 1]

##Elevation and Azimuth input
elevation = 0
pinna = "H"
azimuth = 45
spread = 30

##Azimuth including stereo spread
azimuthL = azimuth - spread;
azimuthR = azimuth + spread;

##Temporary azimuth assignments which will be needed later in convolution
tempAzimuthL = azimuthL
tempAzimuthR = azimuthR

##Swaps the Azimuth so that it encapsulates all 360 degrees
if (azimuthL < 0):
  azimuthL = np.abs(azimuthL)
if (azimuthR < 0):
  azimuthR = np.abs(azimuthR)

filenameL = f"elev{elevation}/{pinna}{elevation}e{azimuthL:03d}a.wav"
filenameR = f"elev{elevation}/{pinna}{elevation}e{azimuthR:03d}a.wav"

hfs, hsig_L = wavfile.read(os.path.join(script_dir, filenameL))
hfs, hsig_R = wavfile.read(os.path.join(script_dir, filenameR))


h_sig_L = resample_poly(hsig_L, fs, hfs)
h_sig_R = resample_poly(hsig_R, fs, hfs)


hL = np.array(h_sig_L)
hR = np.array(h_sig_L)


hL = hL.astype(np.float32)
hL = hL / np.max(np.abs(hL))
hR = hR.astype(np.float32)
hR = hR / np.max(np.abs(hR))


hL_left = hL[:, 0]
hL_right = hL[:, 1]
hR_left = hR[:, 0]
hR_right = hR[:, 1]


if (tempAzimuthL < 0):
  hL_left = hL[:, 1]
  hL_right = hL[:, 0]
if (tempAzimuthR < 0):
  hR_left = hR[:, 1]
  hR_right = hR[:, 0]

## At this point:
##  Input signal L is x_left, R is x_right. System signal L is h_left and R is h_right

##    Convolves input signal L against H left and H right
def left_convolve(chunkleft, chunkright, hL_left, hR_left):

  x_length = len(chunkleft)
  h_length = len(hL_left)
  y_length = x_length + h_length - 1

  y = np.zeros(y_length)

  chunkSize = 1024
  index = 0
  
  for n in range(math.ceil(x_length / chunkSize)):

    leftOne = signal.fftconvolve(chunkleft, hL_left, mode='full')
    leftTwo = signal.fftconvolve(chunkright, hR_left, mode='full')
    leftProduct = leftOne + leftTwo

    y[index : index + len(leftProduct)] += leftProduct

    index = index + chunkSize

  return y


##    Convolves input signal R against H left and H right
def right_convolve(chunkleft, chunkright, hL_right, hR_right):
  x_length = len(chunkright)
  h_length = len(hR_right)
  y_length = x_length + h_length - 1

  y = np.zeros(y_length)

  chunkSize = 1024
  index = 0
  
  for n in range(math.ceil(x_length / chunkSize)):

    rightOne = signal.fftconvolve(chunkleft, hL_right, mode='full')
    rightTwo = signal.fftconvolve(chunkright, hR_right, mode='full')
    rightProduct = rightOne + rightTwo

    y[index : index + len(rightProduct)] += rightProduct

    index = index + chunkSize

  return y





tail_left = np.zeros(len(hL_left) - 1)
tail_right = np.zeros(len(hR_right) - 1)

position = 0

##Callback Structure
def callbackmachine(outdata, frames, time, status):
  global position
  global tail_left
  global tail_right

  chunkleft = x_left[position : position + frames]
  chunkright = x_right[position : position + frames]

  left_out = left_convolve(chunkleft, chunkright, hL_left, hR_left)
  right_out = right_convolve(chunkleft, chunkright, hL_right, hR_right)

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
