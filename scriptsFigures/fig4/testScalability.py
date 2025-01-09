import numpy as np
import sys
sys.path.append('..')
import uafourier
import time

def exponential_kernel(N, lengthscale=15, sigma=1.0):
    x = np.arange(0, N)[:, None]  # Sample points
    x, y = np.meshgrid(x, x)
    # sin_term = np.sin(np.pi * np.abs(x - y))
    cov = sigma**2 * np.exp(-2 * (x-y)**2 / lengthscale**2)
    return cov

def createData(length):
    x = np.arange(length)*1/12
    f = 1#0.05*x#12.0/50*x
    testdata = np.sin(f*x)
    testdata[int(length/2):] = np.sin(3*x[int(length/2):])
    cov = exponential_kernel(len(testdata), 15, 0.5)
    return testdata, cov

timesteps = np.linspace(200, 2000, 10)
timeFourierMean = np.zeros(len(timesteps))
timeFourierMedian = np.zeros(len(timesteps))
timeWaveletMean = np.zeros(len(timesteps))
timeWaveletMedian = np.zeros(len(timesteps))
timeFourierTransform = np.zeros(len(timesteps))
timeWaveletTransform = np.zeros(len(timesteps))
dt = 1
dj = 0.125
for i, t in enumerate(timesteps):
    t = int(t)
    mean, cov = createData(t)

    start = time.time()
    fftMu, fftGamma, fftC = uafourier.ua_fourier(mean, cov)
    timeFourierTransform[i] = time.time()-start

    start = time.time()
    mu_f, sigma_f = uafourier.complex_to_real(fftMu, fftGamma, fftC)
    uafourier.comutePercentiles(mu_f, sigma_f, [0.5])
    timeFourierMedian[i] = time.time()-start

    start = time.time()
    uafourier.energy_spectral_density(fftMu, fftGamma, fftC)
    timeFourierMean[i] = time.time()-start

    start = time.time()
    waveletMu, waveletGamma, waveletC = uafourier.waveletTransform(fftMu, fftGamma, fftC, dt, dj)
    timeWaveletTransform[i] = time.time()-start

    start = time.time()
    uafourier.waveletSpectrum(waveletMu, waveletGamma, waveletC)
    timeWaveletMean[i] = time.time()-start

    start = time.time()
    for j in range(len(waveletMu)):
        uafourier.computePercentilesComplex(waveletMu[j], waveletGamma[j], waveletC[j],[0.5])
    timeWaveletMedian[i] = time.time()-start
    print(str(i) + ": " + str(t) + " in " + str(time.time()-start) + "s")

np.save("timings/fourierMedian.npy", timeFourierMedian)
np.save("timings/fourierMean.npy", timeFourierMean)
np.save("timings/fourierTransform.npy", timeFourierTransform)
np.save("timings/waveletMedian.npy", timeWaveletMedian)
np.save("timings/waveletMean.npy", timeWaveletMean)
np.save("timings/waveletTransform.npy", timeWaveletTransform)
