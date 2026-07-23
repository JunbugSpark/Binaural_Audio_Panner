## Binaural_Audio_Panner
An HRTF-based binaural audio panner that utilizes FFT convolution. The end goal is for this panner to run in real-time, still in progress.

## Phase 1.1 - Direct Time-Domain Convolution
First developed in Google Colab, phase 1.1 contains the initial implementation of the binaural panner, using crude, standard convolution.

Its capabilities
- Takes a mono audio input, and convolves it with a certain HRTF dataset
- The HRTF data is manually set by the user
- HRTF is resampled to match the input signal's sample rate
- Convolution is done separately for the left and right ears
- The resulting outputs are combined into a stereo output

Its limitations
- Standard basic convolution utilizes O(N x M) number of computations (N = length of input, M = length of system), which increases in its computational expensiveness as the audio signal becomes longer
- For a 1-minute audio sample input, the convolution process took 8 minutes
- This was the main motivation for Phase 2.1, where Fast Fourier Transform was utilized.

## Phase 2.1 - Overlap-Add Convolution
Building on Phase 1.1, Phase 2.1 replaces direct time-domain convolution for the overlap-add method, using FFT-based convolution in order to reduce processing time from 8 minutes in phase 1.1 to around 5 seconds in phase 2.1.

Its capabilities
- Splits the input signal into fixed-size chunks (in this instance 1024 samples each)
- Each chunk is convolved with an HRTF sample using FFT-based convolution
- The outputs are summed and overlapped in its original sample position, reconstructing the full convolution result as a sum of all the individual chunks
- Left and right channels are convolved separately, then combined into a stereo signal

Its limitations
- Not in real-time: audio is fully processed before playback, which leads me into phase 2.2
- The next goal is to advance this project towards real-time streaming.

This project uses HRTF data from the MIT KEMAR dataset (Gardner, W. & Martin, K., 1994, MIT Media Lab).
