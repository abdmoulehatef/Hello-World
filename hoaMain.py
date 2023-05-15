import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sounddevice as sd
from hoa import *

matplotlib.use('Qt5Agg')
########################################## Exercise 1.1 ##########################################
theta1 = np.deg2rad(90)
phi1   = np.deg2rad(45)

theta2 = np.deg2rad(90)
phi2   = np.deg2rad(-45)

direction1 = Direction(theta1, phi1)
direction1.plot()

direction2 = Direction(theta2, phi2)
direction2.plot()
plt.show()
########################################## Exercise 1.2 ##########################################
hoaSig = HoaSignal('scene_HOA_O4_N3D_12k.wav')
# todo code here
sigOmni = hoaSig.sigCoeffTime[:, 0]

# Listen to signal
# How to listen
sigNoise = np.random.randn(48000)
sd.play(sigNoise, samplerate=48000)

########################################## Exercise 1.3 ##########################################
#todo code here

########################################## Exercise 1.4 ##########################################
#todo code here

beamformer = Beamformer()
beamformer = beamformer.createBeamformerFromHoaSignal(hoaSig)
signal1 = beamformer.beamformSignal(hoaSig, direction1)
signal2 = beamformer.beamformSignal(hoaSig, direction2)
plt.plot(signal1[0, :])
plt.plot(signal2[0, :])
plt.show()


########################################## Exercise 1.5 ##########################################

# Steered response power map
numAzim = 160
numInc1 = 80
# todo code here
srpMap = SteeredResponsePowerMap(numAzim, numInc1)

# Iterate over frames, calculate and plot steered response power map
frameLength = 2048
frameAdvance = 1024
nFrames = int(np.floor((hoaSig.numSamples - frameLength) / frameAdvance + 1))


def animate(i):
    sampleRange = i * frameAdvance + np.arange(frameLength)

    # Calculate steered response power map for current sample range
    srpMap.generateSrpMap(hoaSig, sampleRange)
    hplot = srpMap.updatePlot()
    hdot = srpMap.markMaximum()

    return [hplot, hdot]


fig = plt.figure()
# Don't use the scientific mode from PyCharm here!!!
anim = animation.FuncAnimation(fig, animate, frames=nFrames, init_func=srpMap.initPlot, interval=1, repeat=False)
plt.show()

########################################## Exercise 1.6 ##########################################
# Create BinauralRenderer for specified HRIR database
renderer = BinauralRenderer.createBinauralRendererFromHoaSignal(hoaSig, 'hrirs_12k.mat')
sigBinaural = renderer.renderSignal(hoaSig)

# Play binaural signal
sd.play(sigBinaural, renderer.fs)
sd.wait()
