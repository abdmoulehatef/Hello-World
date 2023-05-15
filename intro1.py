#!/usr/bin/env python
# coding: utf-8

# 
# # Digital Signalprocessing in Python

# # Session 1: Basics

# ## 1. Properties of speech signals
# 
# **Task 1:** Import the file speech.wav as a numpy array. What sampling rate does the signal have? Play it with the correct sampling rate via the sound card.

# In[1]:


import soundfile as sf
data, sr = sf.read('speech.wav')
print('The sampling rate is: ' + str(sr))


# In[42]:


"""Playing the soundfile"""
import sounddevice as sd
sd.play(data, sr)
status = sd.wait()


# **Task 2:** Extract a section of 8000 samples starting from sample 5500 and plot it so that the x-axis shows the time instead of the index.

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
section = data[5499:5499+8000].copy()
t = np.arange(5499/sr, (5499+8000)/sr, 1/sr)
plt.plot(t,section)


# **Task 3:** Determine minimum and maximum as well as arithmetic mean $\bar{x}$ and squared mean (mean value of $x^2$)
# of the entire speech signal.

# In[92]:


import statistics as stat
minimum = min(data)
print('min = ' + str(minimum))
maximum = max(data)
print('max = ' + str(maximum))
mean = stat.mean(data)
print('mean = ' + str(mean))
squared_mean = stat.mean(data ** 2)
print('squared mean = ' + str(squared_mean))


# **Task 4:** Generate a histogram of the speech signal using the function $\texttt{plt.hist}$. Set the parameters
# so that you receive a plot that covers the value range from $-0.25$ to $+0.25$ with a resolution of $1/500$.

# In[40]:


binwidth = 1/500
plt.hist(data, bins=np.arange(-0.25, 0.25 + binwidth, binwidth))


# 
# **Task 5:** Generate random signals with the same length as the speech signal with each of the functions
# $\texttt{np.random.rand}$, $\texttt{np.random.randn}$ and the unknown function $\texttt{randg}$ that has been imported. Note that $\texttt{randg}$ takes a few seconds.
# 	
# Generate seperate histograms with the same settings as for the speech signal for all of these random signals.
# Modify the value range of the histogram so that it covers the entire signal by making use of the command
# $\texttt{np.min}$ and $\texttt{np.max}$. What is the distribution of the different noise signals? What distribution can
# the speech signal best be described by?

# In[39]:


from numpy import random
import randg

rand_sig1 = np.random.rand(len(data))
rand_sig2 = np.random.randn(len(data))
rand_sig3 = randg.randg(len(data))

plt.subplot(3,1,1)
plt.hist(rand_sig1, bins=np.arange(min(rand_sig1), max(rand_sig1) + binwidth, binwidth))
plt.title('rand signal')
plt.subplot(3,1,2)
plt.hist(rand_sig2, bins=np.arange(min(rand_sig2), max(rand_sig2) + binwidth, binwidth))
plt.title('randn signal')
plt.subplot(3,1,3)
plt.hist(rand_sig3, bins=np.arange(min(rand_sig3), max(rand_sig3) + binwidth, binwidth))
plt.title('randg signal')




# ## 2. Quantization
# 
# **Task 1:** Normalize the input signal to the value range [−1,1] and perform a uniform quanitzation of the speech signal
# from the previous exercise with 8, 6 and 4 bit precision, i. e. 256, 64 and 16 valid steps. The input values are 
# rounded to the closest quantization level in the process. The value range of the quantizer is assumed to be limited to 
# [−1,1]. The quantizer characteristic should be analogous to Figure 1.11.

# In[106]:


data_nrm = -1 + 2*(data - min(data))/(max(data) - min(data))

data_qt16 = 8*data_nrm
data_qt16 = np.around(data_qt16)
data_nrm_qt16 = -1 + 2*(data_qt16 - min(data_qt16))/(max(data_qt16) - min(data_qt16))

data_qt64 = 32*data_nrm 
data_qt64 = np.around(data_qt64)
data_nrm_qt64 = -1 + 2*(data_qt64 - min(data_qt64))/(max(data_qt64) - min(data_qt64))

data_qt256 = 128*data_nrm
data_qt256 = np.around(data_qt256)
data_nrm_qt256 = -1 + 2*(data_qt256 - min(data_qt256))/(max(data_qt256) - min(data_qt256))
print(len(set(data_qt16)))


# **Task 2:** Listen to the quantized signals.

# In[81]:


import sounddevice as sd
sd.play(data_nrm_qt16,sr)
status = sd.wait()
sd.play(data_nrm_qt64,sr)
status = sd.wait()
sd.play(data_nrm_qt256,sr)


# **Task 3:** Plot a section of 500 samples from the original signal and the three quantized signals in a diagram. The 
# selected signal segment should be one where there is “a lot going on”. Use the function $\texttt{plt.step}$ for the quantized signal 
# segments.

# In[91]:


start = 40000
section1 = data_nrm[start:start+500].copy()
section2 = data_nrm_qt16[start:start+500].copy()
section3 = data_nrm_qt64[start:start+500].copy()
section4 = data_nrm_qt256[start:start+500].copy()

x = np.arange(500)

plt.subplot(4,1,1)
plt.step(x,section1)
plt.title('section_nrm')

plt.subplot(4,1,2)
plt.step(x,section2)
plt.title('section_nrm_qt16')

plt.subplot(4,1,3)
plt.step(x,section3)
plt.title('section_nrm_qt64')

plt.subplot(4,1,4)
plt.step(x,section4)
plt.title('section_nrm_qt256')


# **Task 4:** Determine the error signal for the three quantized signals for the entire length of the signals

# In[94]:


err_sig1 = data_nrm - data_nrm_qt16
err_sig2 = data_nrm - data_nrm_qt64
err_sig3 = data_nrm - data_nrm_qt256


# **Task 5:** Generate histograms for the error signals. Plot them together with the histogram of the original signal
# in a single diagram. What do you notice?

# In[111]:


plt.hist(data_nrm,  histtype='step', label='data_nrm')
plt.hist(err_sig1, histtype='step',label='err_sig1')
plt.hist(err_sig2,  histtype='step',label='err_sig2')
plt.hist(err_sig3,  histtype='step',label='err_sig2')
plt.legend
plt.show


# **Task 6:** The signal is normalized to the value range [−1,1]. Set the value range of the quantizer to [−0.01,0.01]
# and apply the quantization with 8 bit once more. Keep in mind that the stepsize changes accordingly. Listen to the result.

# In[178]:


data_nrm_qt256[data_nrm_qt256<-0.01] = -0.01
data_nrm_qt256[data_nrm_qt256>0.01] = 0.01
    
data_qt256_2 = 12800*data_nrm_qt256_2
data_qt256_2 = np.around(data_qt256_2)
data_nrm_qt256_2 = -1 + 2*(data_qt256_2 - min(data_qt256_2))/(max(data_qt256_2) - min(data_qt256_2))
data_nrm_qt256_2 /= 100
sd.play(data_nrm_qt256_2,sr)
status = sd.wait()


# **Task 7:** Determine the indices of the samples that are in the saturation region of the quantizer. Think about how
# you can highlight these positions in a plot of the original signal and create such a plot.

# In[173]:


plt.plot(data)
print(data[data==0.1])
plt.plot(data[data == 0.1], 'ro')


# ## 3. Signal-to-Noise-Ratio
# 
# A measure for the quality of a quantizer is the relation of the energies of the original
# signal and the error signal. This so-called SNR (Signal-to-Noise-Ratio) is defined as (see script).
# 
# Compute the SNR for the four quantizers used in the previous exercise. Again, use
# the speech signal speech.wav as an input signal.

# In[213]:


snr1 = 10*np.log10(np.dot(data_nrm,data_nrm)/np.dot(err_sig1,err_sig1))
snr2 = 10*np.log10(np.dot(data_nrm,data_nrm)/np.dot(err_sig2,err_sig2))
snr3 = 10*np.log10(np.dot(data_nrm,data_nrm)/np.dot(err_sig3,err_sig3))
err_sig4 = data_nrm - data_nrm_qt256_2
snr4 = 10*np.log10(np.dot(data_nrm,data_nrm)/np.dot(err_sig4,err_sig4))
print(snr1)
print(snr2)
print(snr3)
print(snr4)


# ## 4. Amplitude modulation

# **Task 1:** Generate a signal of length t = 1 s with a sampling rate of fs = 32 kHz. The signal should be composed of
# four sine components of the frequencies 50 Hz, 110 Hz, 230 Hz and 600 Hz with the weights 1, 0.4, 0.2 and 0.05. 
# Normalize the signal to the maximum amplitude 1.0, store it in a variable and display it graphically.

# In[222]:


t = np.linspace(0, 1, 32000)
get_ipython().run_line_magic('matplotlib', 'inline')
sig1 = 1*np.sin(2*np.pi*50*t)
sig2 = 0.4*np.sin(2*np.pi*110*t)
sig3 = 0.2*np.sin(2*np.pi*230*t)
sig4 = 0.05*np.sin(2*np.pi*600*t)
signal = sig1 + sig2 + sig3 + sig4 
nrm_sig = signal / max(abs(signal))
plt.plot(t,nrm_sig)







# **Task 2:** Generate a cosine of the same length and sampling rate with the frequency f = 12 kHz and a phase of φ0 = 0.

# In[194]:


cos_sig = 1*np.cos(2*np.pi*12000*t)


# **Task 3:** Perform an amplitude modulation of the signal from point 1 according to equation (1.3). Use each of the
# modulation indices $m = 0.8$ and $m = 1.8$ once. Display the modulated signal, its envelope and the signal to be modulated
# graphically. Use $\texttt{abs(hilbert(amsignal))}$ for the calculation of the envelope where amsignal is the modulated signal. Can
# the signal be received without errors for both modulation indices? The case m > 1 is called overmodulation. Why?

# In[234]:


from scipy.signal import hilbert

a0 = 1 
m1 = 0.8
signal_AM_1 = (a0+m1*nrm_sig) * cos_sig
m2 = 1.8
signal_AM_2 = (a0+m2*nrm_sig) * cos_sig

env_sig1 = abs(hilbert(signal_AM_1))
env_sig2 = abs(hilbert(signal_AM_2))

plt.subplot(5,1,1)
plt.plot(signal,label='input signal')
plt.subplot(5,1,2)
plt.plot(signal_AM_1, label='modulated_signal 1')
plt.subplot(5,1,3)
plt.plot(signal_AM_2, label='modulated_signal 2')
plt.subplot(5,1,4)
plt.plot(env_sig1 , label='envelopped_signal 1')
plt.subplot(5,1,5)
plt.plot(env_sig2, label='envelopped_signal m1')
plt.legend()


# **Task 4:** Demodulate the signals generated in the point before. To do so, first compute the corresponding analytical
# signal using the Hilbert transform (see function $\texttt{hilbert}$).Then, perform a frequency shift to the baseband by
# multiplying with $e^{-\text{j}\omega t}$ in the time domain. Finally, take the absolute value of the signal, 11
# subtract the direct component $\frac{1}{a_0}$ and scale it with $\frac{1}{a_0 m}$.
# 
# Keep in mind that the function $\texttt{hilbert}$ returns the analytical signal and not just the hilbert transform.

# In[244]:


analytic_signal1 = hilbert(env_sig1) 
analytic_signal2 = hilbert(env_sig2) 

dem_sig1 = ( np.abs(analytic_signal1 * np.exp(-2j*np.pi*12000*t)) - a0 ) / (a0*m1)
dem_sig2 = ( np.abs(analytic_signal2 * np.exp(-2j*np.pi*12000*t)) - a0 ) / (a0*m2)
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(nrm_sig, label='nrm_sig')
plt.plot(dem_sig1, linestyle = 'dashed', label='dem_sig1')
plt.plot(dem_sig2, linestyle = 'dashed', label='dem_sig2')
plt.legend()


# In[ ]:




