import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat


def nlms4echokomp(x, g, noise, alpha, mh):
    """ The MATLAB function 'nlms4echokomp' simulates a system for acoustic echo compensation using NLMS algorithm
    :param x:       Input speech signal from far speaker
    :param g:       Impluse response of the simulated room
    :param noise:   Speech signal from the near speaker and the background noise(s + n)
    :param alpha:   Step size for the NLMS algorithm
    :param mh:      Length of the compensation filter

    :return s_diff:  relative system distance in dB
    :return err:    error signal e(k)
    :return x_hat:  output signal of the compensation filter
    :return x_tilde:acoustic echo of far speakers
    """

    # Initialization of all the variables
    lx = len(x)  # Length of the input sequence
    mg = len(g)  # Length of the room impulse response(RIR)
    if mh > mg:
        mh = mg
        import warnings
        warnings.warn('The compensation filter is shortened to fit the length of RIR!', UserWarning)

    # Vectors are initialized to zero vectors.
    x_tilde = np.zeros(lx - mg)
    x_hat = x_tilde.copy()
    err = x_tilde.copy()
    s_diff = x_tilde.copy()
    ERLE = np.ones(lx - mg - 200)
    h = np.zeros(mg)

    # Realization of NLMS algorithm
    k = 0
    for index in range(mg, lx):
        # Extract the last mg values(including the current value) from the
        # input speech signal x, where x(i) represents the current value.
        # todo your code
        x_block = x[k:index]

        # Filtering the input speech signal using room impulse response and adaptive filter. Please note that you don't
        # need to implement the complete filtering here. A simple vector manipulation would be enough here
        # todo your code:
        x_tilde[k] = np.dot(g.T, x_block) + noise[k]
        x_hat[k] = np.dot(h.T,x_block)
        if mh != mg:
           # x_hat[k] = np.dot(h.T[:mh], x_block[:mh])
           # # x_hat[index-(mg-mh):index] = 0
           # if k>index-(mg-mh) & k<index:
           #  x_hat[k] = 0
            h[mh:mg]=0


        # Calculating the estimated error signal
        # todo your code
        err[k] = x_tilde[k] - x_hat[k]

        # Updating the filter
        # todo your code
        if np.dot(x_block.T, x_block) != 0:
            h = h + x_block * err[k] * alpha / np.dot(x_block.T, x_block)
        # Calculating the relative system distance
        # todo your code
        s_diff[k] = np.dot((g- h).T, g - h) / np.dot(g.T, g)

        if k > 199:
            if np.mean(x_tilde[k - 200:k] ** 2) !=0 and np.mean((x_tilde[k - 200:k] - x_hat[k - 200:k]) ** 2) != 0:
                ERLE[k - 200] = np.mean(x_tilde[k - 200:k] ** 2) / np.mean((x_tilde[k - 200:k] - x_hat[k - 200:k]) ** 2)

        k = k + 1  # time index

    # todo your code
    s_diff = 10 * np.log10(s_diff[:k]).T
    ERLE = 10 * np.log10(ERLE).T

    # Calculating the relative system distance in dB
    return ERLE, s_diff, err, x_hat, x_tilde


# switch between exercises
exercise = 7 # choose between 1-7

f = np.load('echocomp.npz')
g = [f['g1'], f['g2'], f['g3']]
s = f['s']

# Generation of default values
alpha = 0.1  # Step size for NLMS

ls = len(s)  # Length of the speech signal
n0 = np.sqrt(0.16) * np.random.randn(ls)  # White Noise
s = s / np.sqrt(s.T.dot(s)) * np.sqrt(n0.T.dot(n0))  # Number of curves in each plot (should not be changed)
vn = 3  # number of curves
noise = [np.zeros(ls, ) for i in range(vn)]  # no disturbance by noise
alphas = [alpha for i in range(vn)]  # Step size factor for different exercises
mh = len(g[0]) * np.ones(vn, dtype=int)  # Length of the compensation filter
#mh = [len(g[0]), len(g[1]), len(g[2])]  # Length of the compensation
x = [n0.copy() for i in range(vn)]  # white noise as input signal

# In the following part, the matrices and vectors must be adjusted to
# meet the requirement for different exercises
# (Exercise 1 can be simulated using only the initialized values above)


if exercise == 2:
    # Only the value of input speech signal need to be changed. All the other
    # vectors and parameters should not be modified

    x[0] = s  # Speech signal
    # todo your code
    x[1] = n0 # white noise

    b = [1]
    a = [1, -0.5]
    x[2] = signal.lfilter(b, a, x[1])

    g = [g[0], g[0], g[0]]

    leg = ('Speech', 'white noise', 'colorful noise')
    title = 'Influence of input signals with g=g0 and noise=false'
elif exercise == 3:
    x[0] = n0  # white noise
    x[1] = n0  # white noise
    x[2] = n0  # white noise
    noise[0] = np.sqrt(0.) * np.random.randn(ls)  # white noise
    noise[1] = np.sqrt(0.001) * np.random.randn(ls)
    noise[2] = np.sqrt(0.01) * np.random.randn(ls)
    g = [g[0], g[0], g[0]]
    leg = ('n[0] = 0', 'n[1] = 0.001', 'n[2] = 0.01')
    title = 'Influence of noises with x=n0, g=g0'
    pass
elif exercise == 4:
    x[0] = s  # Speech signal
    x[1] = s  # Speech signal
    x[2] = s  # Speech signal
    noise[0] = np.sqrt(0.) * np.random.randn(ls)  # white noise
    noise[1] = np.sqrt(0.001) * np.random.randn(ls)
    noise[2] = np.sqrt(0.01) * np.random.randn(ls)
    g = [g[0], g[0], g[0]]
    leg = ('n[0] = 0', 'n[1] = 0.001', 'n[2] = 0.01')
    title = 'Influence of noises on speech signal x=s, g=g0 and noise=true'
    pass

elif exercise == 5:
    x[0] = n0  # White noise
    x[1] = n0
    x[2] = n0
    noise[0] = np.sqrt(0.01) * np.random.randn(ls) # white noise
    noise[1] = np.sqrt(0.01) * np.random.randn(ls)
    noise[2] = np.sqrt(0.01) * np.random.randn(ls)
    g = [g[0], g[0], g[0]]
    alphas = [0.1, 0.5, 1.0]
    leg = ('alpha = 0.1', 'alpha = 0.5', 'alpha = 1.0')
    title = 'Influence of stepsizes alpha with x=s, g=g0 and noise=true'
    pass

elif exercise == 6:
    x[0] = n0  # White noise
    x[1] = n0
    x[2] = n0
    noise[0] = np.sqrt(0.001) * np.random.randn(ls) # white noise
    noise[1] = np.sqrt(0.001) * np.random.randn(ls)
    noise[2] = np.sqrt(0.001) * np.random.randn(ls)
    mh = [len(g[0])-10, len(g[0])-30, len(g[0])-60]
    g = [g[0], g[0], g[0]]
    leg = ('mh = mg-10', 'mh = mg-30', 'mh = mg-60')
    title = 'Influence of mh'
    pass

elif exercise == 7:
    x[0] = n0  # White noise
    x[1] = n0
    x[2] = n0
    noise = [np.zeros(ls, ) for i in range(vn)]  # no disturbance by noise

    g = [g[0], g[1], g[2]]
    mh = [len(g[0]), len(g[1]), len(g[2])]
    leg = ('g[0]', 'g[1]', 'g[2]')
    title = 'Influence of different lengths g with x=n0 and Noise=false'
    pass
# There should be appropriate legends and axis labels in each figure!
if exercise == 1:
    ERLE, s_diff, e, x_h, x_t = nlms4echokomp(n0, g[0], np.zeros(ls), alpha, 200)
    t = np.arange(0., len(x_h) / 8000, 1 / 8000)
    plt.figure(0)
    plt.subplot(311)
    t0 = np.arange(0., len(n0) / 8000, 1 / 8000)
    plt.plot(t0, n0)
    plt.xlabel('t [s]')
    plt.ylabel('n0')
    plt.subplot(312)
    plt.plot(t, x_h)
    plt.xlabel('t [s]')
    plt.ylabel('x_h')
    plt.subplot(313)
    plt.plot(t, x_t, label='x_t')
    plt.plot(t, e, label='error')
    plt.legend(loc='best')
    plt.xlabel('t [s]')
    plt.ylabel('x_t/e')
    plt.show()
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, s_diff)
    plt.ylabel('s_diff [dB]')
    plt.subplot(212)
    t2 = np.arange(0., len(ERLE) / 8000, 1 / 8000)
    plt.plot(t2, ERLE)
    plt.ylabel('ERLE [dB]')
    plt.xlabel('t [s]')
    plt.show()

else:
    for i in range(vn):
        # 3 system distances with different parameters are calculated here
        # The input variables of 'nlms4echokomp' must be adapted according
        # to different exercises.

        ERLE, s_diff, e, x_h, x_t = nlms4echokomp(x[i], g[i], noise[i], alphas[i], mh[i])
        plt.plot(s_diff, label=leg[i])

    plt.title('Exercise ' + str(exercise) + ': ' + title)
    plt.xlabel('k')
    plt.ylabel('D(k) [dB]')
    plt.grid(True)
    plt.legend()
    plt.show()
