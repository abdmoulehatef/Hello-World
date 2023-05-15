import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile, loadmat
from scipy.special import sph_harm
from scipy.signal import oaconvolve
import math

class Direction:

    def __init__(self, theta, phi):
        self._theta = np.array(theta).flatten()
        self._phi = np.array(phi).flatten()

    def getThetaInRadians(self):
        return self._theta

    def getPhiInRadians(self):
        return self._phi

    def getThetaInDegrees(self):
        return self._theta * 180 / np.pi

    def getPhiInDegrees(self):
        return self._phi * 180 / np.pi

    def plot(self):
        plt.scatter(self._phi, self._theta)

class HoaSignal:

    def __init__(self, sFilename=None):
        if sFilename is None:
            sFilename = 'signal_HOA_O4_N3D_12k.wav'

        # Load file
        self.fs, self.sigCoeffTime = wavfile.read(sFilename)

        # Save properties
        # todo code here
        xi = np.array(self.sigCoeffTime).shape[1]
        self.hoaOrder = int(-1 + np.sqrt(xi))
        self.numSamples = np.array(self.sigCoeffTime).shape[0]


class Beamformer:

    def __init__(self, hoaOrder=None):
        if hoaOrder is not None:
            self.hoaOrder = hoaOrder

    @classmethod
    def createBeamformerFromHoaSignal(cls, hoaSignal):
        return cls(hoaSignal.hoaOrder)

    def beamformSignal(self, hoaSignal, direction, sampleRange=None):
        # Q = hoaSignal.numSamples
        # N = hoaSignal.hoaOrder
        # xi_dim = N**2+2*N+1
        Y = self.__createSphericalHarmonicsMatrix(direction.getThetaInRadians(), direction.getPhiInRadians())
        if sampleRange is None:
            return Y @ hoaSignal.sigCoeffTime.T
        else:
            return Y@hoaSignal.sigCoeffTime[sampleRange, :].T
        # for q in range(Q):
        #     for n in range(N+1):
        #         for m in range(-n, n+1):
        #             xi = n**2 + n + m + 1
        #             sq_root = np.sqrt((2*n+1)/(4*np.pi)*math.factorial(n-m)/math.factorial(n+m))
        #             Y[q, xi] = sq_root*np.exp()

    def __createSphericalHarmonicsMatrixFromDirection(self):
        """
        Creates the Spherical Harmonics Matrix for the directions given in the Direction object self.direction
        Parameter handling.
        :return:
        """

        assert ~isinstance(self, Direction)

        return self.__createSphericalHarmonicsMatrix(self.direction.getThetaInRadians(),
                                                     self.direction.getPhiInRadians())

    def __createSphericalHarmonicsMatrix(self, theta, phi):
        """
        creates a spherical harmonics matrix

        :param theta: Column vector of inclination angles in radians
        :param phi: Column vector of azimuth angles in radians
        :return: Y =
                                    n = 0                      n = 1                                    n = N
                                /------------\  /---------------------------------------------\  ... --------------\

                   gamma_1   /  Y_0^0(gamma_1)  Y_1^-1(gamma_1)  Y_1^0(gamma_1)  Y_1^1(gamma_1)  ...  Y_N^N(gamma_1)  \
                   gamma_2   |  Y_0^0(gamma_2)  Y_1^-1(gamma_2)  Y_1^0(gamma_1)  Y_1^1(gamma_1)  ...  Y_N^N(gamma_2)  |
                   gamma_3   |  Y_0^0(gamma_3)  Y_1^-1(gamma_3)  Y_1^0(gamma_1)  Y_1^1(gamma_1)  ...  Y_N^N(gamma_3)  |
                    ...     |     ...             ...              ...             ...          ...     ...          |
                   gamma_Q   \  Y_0^0(gamma_Q)  Y_1^-1(gamma_Q)  Y_1^0(gamma_1)  Y_1^1(gamma_1)  ...  Y_N^N(gamma_Q)  /
        """

        assert (len(theta) == len(phi))
        N = self.hoaOrder
        K = (N + 1) ** 2
        Q = len(theta)

        # Different mapping than usual (https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html)
        azimRad = phi
        inclRad = theta

        alln = np.array([i for i in range(0, N + 1) for j in range(0, 2 * i + 1)])
        allm = np.array([j - i for i in range(0, N + 1) for j in range(0, 2 * i + 1)])

        # repeat orders and degrees for each point q such that all inputs have Q*K elements
        m = np.tile(allm, Q)
        n = np.tile(alln, Q)
        az = np.repeat(azimRad, K)
        incl = np.repeat(inclRad, K)

        Y_cmplx = sph_harm(m, n, az, incl)

        # convert to real SHs
        Y_real = Y_cmplx
        Y_real[m > 0] = np.array([-1.0]) ** m[m > 0] * np.sqrt(2) * np.real(Y_real[m > 0])
        Y_real[m < 0] = np.array([-1.0]) ** m[m < 0] * np.sqrt(2) * np.imag(Y_real[m < 0])
        Y_real[np.logical_and(m < 0, (m % 2) == 0)] = -Y_real[np.logical_and(m < 0, (m % 2) == 0)]
        Y_real = np.reshape(Y_real, [Q, K])
        Y = np.real(Y_real)  # make sure this is really real.

        return Y


# todo code here
class SteeredResponsePowerMap(Beamformer):

    def __init__(self, numAzimuths, numInclinations, hoaOrder=4, fs=12e3):
        super(SteeredResponsePowerMap, self).__init__(hoaOrder)

        self.fs = fs

        self.inclinationVec = np.linspace(0, np.pi, numInclinations)
        self.azimuthVec = np.linspace(-np.pi, np.pi, numAzimuths)

        self.numAzimuths = numAzimuths
        self.numInclinations = numInclinations

        self.azimuthGrid, self.inclinationGrid = np.meshgrid(self.azimuthVec, self.inclinationVec)
        self.direction = Direction(self.inclinationGrid.flatten(), self.azimuthGrid.flatten())
        self.numDirection = numInclinations * numAzimuths

    @classmethod
    def createSteeredResponsePowerMapFromHoaSignal(cls, hoaSignal, numAzimuths, numInclinations):
        return cls(numAzimuths, numInclinations, hoaSignal.hoaOrder, hoaSignal.fs)

    def generateSrpMap(self, hoaSignal, sampleRange=None):
        # Default parameter
        if sampleRange is None:
            sampleRange = np.arange(0, hoaSignal.numSamples)

        self.idxLastSample = sampleRange[-1]
        self.numSamples = hoaSignal.numSamples

        signal = self.beamformSignal(hoaSignal, self.direction, sampleRange)
        signal_power = np.sum(signal**2, axis=1)

        self.signal_power = signal_power

        return 0

    def initPlot(self):
        self.hplot = plt.imshow(np.random.rand(self.numInclinations, self.numAzimuths), interpolation='none',
                                cmap=plt.get_cmap('rainbow'), extent=np.rad2deg([-np.pi, np.pi, np.pi, 0]))

        self.hDot = plt.scatter(0, 0)
        plt.xlabel('azimuth [degree]')
        plt.ylabel('inclination [degree]')

        return [self.hplot, self.hDot]

    def updatePlot(self):
        # todo code here
        dataInDb = 10*np.log10(self.signal_power)
        dataInDb = np.reshape(dataInDb, (self.numInclinations, self.numAzimuths))

        self.hplot.set_array(dataInDb)
        self.hplot.set_clim(np.min(dataInDb), np.max(dataInDb))

        plt.title('steered response power map\n time ' + "{:.2f}".format(self.idxLastSample / self.fs) + 's / ' + str(
            self.numSamples / self.fs) + 's')
        return [self.hplot]

    def markMaximum(self):
        # todo code here

        return self.hDot


class BinauralRenderer(Beamformer):

    def __init__(self, sFilenameHrir='hrirs_12k.mat', hoaOrder=4, fs=12e3):
        super(BinauralRenderer, self).__init__(hoaOrder)

        # todo code here
        self.hrirs = loadmat(sFilenameHrir)
        self.fs = fs

    @classmethod
    def createBinauralRendererFromHoaSignal(cls, hoaSignal, sFilenameHrir=None):
        # todo code here
        if sFilenameHrir is None:
            return cls(hoaOrder=hoaSignal.hoaOrder, fs=hoaSignal.fs)
        else:
            return cls(sFilenameHrir=sFilenameHrir, hoaOrder=hoaSignal.hoaOrder, fs=hoaSignal.fs)

    def renderSignal(self, hoaSignal):
        # todo code here
        beamformer = Beamformer.createBeamformerFromHoaSignal(hoaSignal)
        signal = beamformer.beamformSignal(hoaSignal, Direction(self.hrirs['theta'], self.hrirs['phi']))

        signalForLeftEar = oaconvolve(signal[0], self.hrirs['hrirs'][0, :, 0])
        signalForRightEar = oaconvolve(signal[0], self.hrirs['hrirs'][0, :, 1])

        for i in range(1, 25):
            signalForLeftEar += oaconvolve(signal[i], self.hrirs['hrirs'][i, :, 0])
            signalForRightEar += oaconvolve(signal[i], self.hrirs['hrirs'][i, :, 1])

        signalForLeftEar = np.reshape(signalForLeftEar, (1, -1))
        signalForRightEar = np.reshape(signalForRightEar, (1, -1))

        return np.vstack((signalForLeftEar, signalForRightEar)).T
