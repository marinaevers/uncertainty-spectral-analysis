import math
import numpy as np
from skimage import color
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
from chi2comb import chi2comb_cdf, ChiSquared
from joblib import Memory
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

EPS = 10e-8
CHI2COMBTOL = 1e-3
memory = Memory("pdf-cachedir", verbose=0)


# Transformation matrix
def fourier(i, j, N):
    return np.exp(-2j*math.pi*i*j/N)

def ua_fourier(mu, Sigma):
    N = len(mu)
    W = 1/np.sqrt(N)*np.array([[fourier(i, j, N) for j in range(N)] for i in range(N)], dtype=complex)
    fftMu = np.dot(W, mu)
    fftGamma = np.dot(W, np.dot(Sigma, W.conj().T))
    fftC = np.dot(W, np.dot(Sigma, W.T))
    return fftMu, fftGamma, fftC

# Make use of the symmetry for the Fourier transform for real data,
# Rewrite complex distribution as real random variable X where the first half of the entries corresponds to the real part
# and the second half to the imaginary part
def complex_to_real(mu, Gamma, C):
    N = len(mu)
    muC = np.copy(mu)
    mu_X = np.real(muC)
    mu_X[int(N/2):] = np.imag(muC)[:int(N/2)]
    Sigma = 0.5*np.real(Gamma+C)
    Sigma[int(N/2):, int(N/2):] = 0.5*np.real(Gamma-C)[:int(N/2), :int(N/2)]
    Sigma[int(N/2):, :int(N/2)] = 0.5*np.imag(Gamma+C)[:int(N/2), :int(N/2)]
    Sigma[:int(N/2), int(N/2):] = 0.5*np.imag(C-Gamma)[:int(N/2), :int(N/2)]
    # print("Check matrix")
    # print(Sigma[int(N / 2):, :int(N / 2)][0,0])
    # print(Sigma[:int(N/2), int(N/2):][0,0])
    return mu_X, Sigma

def complex_to_real_block(mu, Gamma, C):
    N = len(mu)
    muC = np.copy(mu)
    mu_X = np.real(muC)
    mu_X[int(3*N/2):] = np.imag(muC)[N:int(3*N/2)]
    Sigma = 0.5*np.real(Gamma+C)
    Sigma[int(3*N/2):, int(3*N/2):] = 0.5*np.real(Gamma-C)[N:int(3*N/2), N:int(3*N/2)]
    Sigma[int(3*N/2):, N:int(3*N/2)] = 0.5*np.imag(Gamma+C)[N:int(3*N/2), N:int(3*N/2)]
    Sigma[N:int(3*N/2), int(3*N/2):] = 0.5*np.imag(C-Gamma)[N:int(3*N/2), N:int(3*N/2)]
    return mu_X, Sigma

def covariance_time_frequency(sigma):
    N = sigma.shape[0]
    W = 1 / np.sqrt(N) * np.array([[fourier(i, j, N) for j in range(N)] for i in range(N)], dtype=complex)
    cov_t_f_pseudo_cov = np.dot(sigma, W)
    cov_t_f_cov = np.dot(sigma, W.conjugate())
    cov_t_f_re = 0.5 * np.real(cov_t_f_cov + cov_t_f_pseudo_cov)[:,:int(0.5*N)]
    cov_t_f_im = 0.5 * np.imag(cov_t_f_pseudo_cov - cov_t_f_cov)[:,:int(0.5*N)]
    cov_tf = np.concatenate((cov_t_f_re, cov_t_f_im), axis=1)
    return cov_tf

def covariance_frequency_spectrum(mu_f, sigma_f):
    N = len(mu_f)
    cov_f_s = 2 * mu_f[:int(N / 2)].reshape(int(N / 2), 1) * sigma_f[:int(N / 2)] + 2 * mu_f[int(N / 2):].reshape(
        int(N / 2), 1) * sigma_f[int(N / 2):]
    return cov_f_s.T

def covariance_time_spectrum(mu_f, mu, sigma):
    N = len(mu_f)
    W = 1 / np.sqrt(N) * np.array([[fourier(i, j, N) for j in range(N)] for i in range(N)], dtype=complex)
    cov_t_f_pseudo_cov = np.dot(sigma, W)
    cov_t_f_cov = np.dot(sigma, W.conjugate())
    cov_t_f_re = 0.5 * np.real(cov_t_f_cov + cov_t_f_pseudo_cov)
    cov_t_f_im = 0.5 * np.imag(cov_t_f_pseudo_cov - cov_t_f_cov)
    cov_s_t = 2 * mu_f[:int(N / 2)].reshape(int(N / 2), 1) * cov_t_f_re[:, :int(N / 2)].T + 2 * mu_f[int(N / 2):].reshape(int(N / 2), 1) * cov_t_f_im[:, :int(N / 2)].T
    return cov_s_t.T

def covariance_density(mu, Sigma):
    N = len(mu)
    cov1 = 2*Sigma[:int(N/2),:int(N/2)]**2+4*np.outer(mu[:int(N/2)],mu[:int(N/2)])*Sigma[:int(N/2),:int(N/2)]
    cov2 = 2*Sigma[:int(N/2),int(N/2):]**2+4*np.outer(mu[:int(N/2)],mu[int(N/2):])*Sigma[:int(N/2),int(N/2):]
    cov3 = 2*Sigma[int(N/2):,:int(N/2)]**2+4*np.outer(mu[int(N/2):],mu[:int(N/2)])*Sigma[int(N/2):,:int(N/2)]
    cov4 = 2*Sigma[int(N/2):,int(N/2):]**2+4*np.outer(mu[int(N/2):],mu[int(N/2):])*Sigma[int(N/2):,int(N/2):]
    return cov1 + cov2 + cov3 + cov4

def create_wavelet(s, w, dt):
    if w < 0:
        print("Zero?!" + str(w))
        return 0
    w_0 = 6 # I have no idea why, but probably reference 4 would tell me
    phi_0 = 1.0/math.pow(math.pi, 0.25)*np.exp(-(s*w-w_0)**2/2)
    phi = np.sqrt(2*math.pi*s/dt)*phi_0
    return phi

def waveletTransform(fftMu, fftGamma, fftC, dt, dj):
    s_0 = 2 * dt
    J = int(math.log2(len(fftMu) * dt / s_0) / dj)
    waveletMu = np.zeros((J, len(fftMu)),dtype='complex')
    waveletGamma = np.zeros((J, len(fftMu), len(fftMu)),dtype='complex')
    waveletC = np.zeros((J, len(fftMu), len(fftMu)),dtype='complex')
    for i in range(J):
        waveletMu[i], waveletGamma[i], waveletC[i] = wavelet(fftMu, fftGamma, fftC, s_0 * math.pow(2, i * dj), dt)
    return waveletMu, waveletGamma, waveletC

def waveletSpectrum(waveletMu, waveletGamma, waveletC):
    idx = range(len(waveletMu[0]))
    var_re_re = 0.5 * np.real(waveletGamma[:,idx,idx] + waveletC[:,idx,idx])
    var_im_im = 0.5 * np.real(waveletGamma[:,idx,idx] - waveletC[:,idx,idx])
    var_re_im = 0.5 * np.imag(waveletGamma[:,idx,idx] + waveletC[:,idx,idx])
    spectrumMu = np.real(waveletMu) ** 2 + np.imag(waveletMu) ** 2 + var_re_re + var_im_im
    var_re_re_square = 2 * var_re_re ** 2 + 4 * np.real(waveletMu) ** 2 * var_re_re
    var_im_im_square = 2 * var_im_im ** 2 + 4 * np.imag(waveletMu) ** 2 * var_im_im
    var_re_im_square = 2 * var_re_im ** 2 + 4 * np.real(waveletMu) * np.imag(waveletMu) * var_re_im
    var_im_re_square = var_re_im_square
    spectrumVar = var_re_re_square + var_im_im_square + var_re_im_square + var_im_re_square
    return spectrumMu, spectrumVar

def wk(k, N, dt):
    if k <= N/2:
        return 2*math.pi*k/(N*dt)
    else:
        return -2*math.pi*k/(N*dt)

def inverse_fourier(i, j, N):
    return np.exp(2j*math.pi*i*j/N)

def wavelet(mu, gamma, C, s, dt=1):
    N = len(mu)
    A = 1/np.sqrt(N)*np.array([[create_wavelet(s, wk(j, N, dt), dt).conjugate()*inverse_fourier(i, j, N) for j in range(int(N/2))] for i in range(N)], dtype=complex)
    waveletMu = np.dot(A, mu[:int(N/2)])
    waveletGamma = np.dot(A, np.dot(gamma[:int(N/2), :int(N/2)], A.conj().T))
    waveletC = np.dot(A, np.dot(C[:int(N/2), :int(N/2)], A.T))
    return waveletMu, waveletGamma, waveletC

def energy_spectral_density(mu, Gamma, C):
    # Rewrite to real variable and omit duplicate entries
    N = len(mu)
    mu_X, Sigma = complex_to_real(mu, Gamma, C)
    mu_Square = mu_X*mu_X
    mu_density = np.diag(Sigma)[:int(N/2)]+np.diag(Sigma)[int(N/2):]+mu_Square[:int(N/2)]+mu_Square[int(N/2):]#[:int(len(mu_Square)/2)]+mu_Square[int(len(mu_Square)/2):]
    #mu_density = mu_Square[:int(N/2)]+mu_Square[int(N/2):]#[:int(len(mu_Square)/2)]+mu_Square[int(len(mu_Square)/2):]
    cov_density = covariance_density(mu_X, Sigma)
    return mu_density, cov_density

def ellipseAngle(lambda1, a, b, c):
    if b == 0 and a >= c:
        return 0
    elif b == 0:
        return np.pi/2
    else:
        return np.arctan2(lambda1-a, b)

# Percentiles
# def integrand(y, x, l1, l2, b1, b2, prefactor):
#   #if y*(x-y) < EPS:
#   #    return 0
#   return prefactor*1/np.sqrt(y*(x-y))*np.exp(-0.5*y*(1/l1-1/l2))*np.cosh(np.sqrt(b1**2*y/l1))*np.cosh(np.sqrt(b2**2*(x-y)/l2))

def f(x, lambd):
    # res = np.zeros(lambd.shape)
    # mask = x>0
    # a = 1/(np.sqrt(2*np.pi*x[mask]))
    # b = np.exp(-0.5*(x[mask]+lambd[mask]))
    # c = np.cosh(np.sqrt(lambd[mask]*x[mask]))
    # res[mask] = a*b*c
    # return res
    return 1/(np.sqrt(2*np.pi*x))*np.exp(-0.5*(x+lambd))*np.cosh(np.sqrt(lambd*x))

def integrand(y, x, l1, l2, b1, b2):
    if np.all(x == y):
        return 0*l1
    return 1/(l1*l2)*f(y/l1, b1**2)*f((x-y)/l2,b2**2)
# def pdf(x, l1, l2, b1, b2):
#     EPS = 0
#     if x <= 0:
#         return 0
#     else:
#         # print(np.sqrt(b1**2/l1))
#         # print(np.sqrt(b2**2/l2))
#         # print(b1**2+b2**2+x/l2)
#         # print(np.exp(-0.5*(b1**2+b2**2+x/l2)))
#         # print((2*np.pi*np.sqrt(l1*l2)))
#         # print(np.exp(-0.5*(b1**2+b2**2+x/l2))/(2*np.pi*np.sqrt(l1*l2)))
#         prefactor = np.exp(-0.5*(b1**2+b2**2+x/l2))/(2*np.pi*np.sqrt(l1*l2))
#         #print(prefactor)
#     if x < EPS or prefactor < EPS:
#         return 0
#     result, error = integrate.quad(integrand, EPS, x, args=(x, l1, l2, b1, b2, prefactor))
#     #print("Result for " + str(x) + ": " + str(result))
#     return result

#@memory.cache
# def pdf(x, l1, l2, b1, b2):
#   if x <= 0:
#     return 0
#   elif l2 < EPS:
#       return 1/l1*f(x/l1, b1**2)
#   else:
#     result, error = integrate.quad(integrand, 0, x, args=(x, l1, l2, b1, b2))
#     return result

def pdf(x, l1, l2, b1, b2):
    if isinstance(x, float) and x <= 0:
        return np.zeros(l1.shape)
    if isinstance(l1, np.ndarray):
        result = np.zeros_like(l1)
        if np.any(l2 < EPS):
            #print("Here")
            mask2 = l2 < EPS
            mask3 = np.logical_not(mask2)

            result[mask2] = 1 / l1[mask2] * f(x / l1[mask2], b1[mask2] ** 2)
            # result[mask3] = integrate.quad_vec(lambda y: integrand(y,x,l1[mask3], l2[mask3], b1[mask3], b2[mask3]), 0, x)[0]#, args=(x, l1[mask3], l2[mask3], b1[mask3], b2[mask3]))[0]
            result[mask3] = integrate.quad_vec(lambda y: x*integrand(x*y,x,l1[mask3], l2[mask3], b1[mask3], b2[mask3]), 0, 1)[0]#, args=(x, l1[mask3], l2[mask3], b1[mask3], b2[mask3]))[0]
        else:
            result = integrate.quad_vec(lambda y: x*integrand(x*y,x,l1, l2, b1, b2), 0, 1)[0]#, epsabs=1e-10)[0]#, epsrel=1e-3)[0]#, args=(x, l1, l2, b1, b2))[0]
        return result
    else:
        if x <= 0:
            return 0
        elif l2 < EPS:
            return 1 / l1 * f(x / l1, b1 ** 2)
        else:
            result, _ = integrate.quad_vec(integrand, 0, x, args=(x, l1, l2, b1, b2))
            return result

def cdf(x, l1, l2, b1, b2):
  #return integrate.quad(pdf, 0, x, args=(l1, l2, b1, b2))[0]
  gcoef = 0
  ncents = [b1**2, b2**2]
  dofs = [1, 1]
  coefs = [l1, l2]
  chi2s = [ChiSquared(coefs[i], ncents[i], dofs[i]) for i in range(2)]
  r, _, _ = chi2comb_cdf(x, chi2s, gcoef, atol=CHI2COMBTOL)
  if r < 0:
      print("wrong " + str(r))
      print(ncents)
      print(coefs)
  return r

def cdf_vec(x, l1, l2, b1, b2):
  #return integrate.quad(pdf, 0, x, args=(l1, l2, b1, b2))[0]
  gcoef = 0
  r = [chi2comb_cdf(x, [ChiSquared(l1[i], b1[i]**2, 1), ChiSquared(l2[i], b2[i]**2, 1)], gcoef, atol=CHI2COMBTOL)[0] for i in range(len(l1))]
  r = np.array(r)
  if np.any(r>1):
      a = np.argwhere(r>1)[0]
      print(r[a])
      print(l1[a])
      print(l2[a])
  # r = [chi2comb_cdf(x, [ChiSquared(l1[i], b1[i]**2, 1), ChiSquared(l2[i], b2[i]**2, 1)], gcoef, atol=1e-3)[0] for i in range(len(l1))]
  return np.array(r)

def cdfSingular(x, l, b, c):
    #print("Singular pdf")
    #print(c)
    gcoef = 0
    # chi2s = [ChiSquared(l, b, 1)]
    chi2s = [ChiSquared(l, (b/l)**2, 1)]
    r, _, _ = chi2comb_cdf(x-c, chi2s, gcoef)
    return r

def logPdf(x, l1,b1,l2,b2):
    pdfVal = pdf(x,l1,l2,b1,b2)
    pdfVal[pdfVal<=0] = 1
    # if pdfVal <= 0:
    #     #print(pdf)
    #     return 0
    return pdfVal*np.log(pdfVal)

def klDivergence(l1, l2, b1, b2, p, mu):
    # probability to surpass 95% percentile of noise
    res = 1 - cdf_vec(p, l1, l2, b1, b2)#integrate.quad_vec(lambda x: pdf(x+p, l1,l2, b1,b2), 0, 1000)[0]#, epsabs=1e-10)[0]
    return res
    integral, _ = integrate.quad_vec(logPdf, 0, 100, args=(l1, b1, l2, b2), epsabs=1e-3)#, epsrel=1e-3)#, epsrel=0.01)
    return integral + np.log(p) + 1 / p * mu

# def klDivergence(l1,l2,b1,b2,p,mu):
#     # x_samples = np.arange(1000)*0.01
#     # samples = [logPdf(x, l1, b1,l2, b1) for x in x_samples]
#     # return integrate.simpson(samples, x=x_samples)
#     return integrate.quad_vec(lambda x: logPdf(x, l1, b1, l2, b2), 0, 100)[0] + math.log(p) + 1 / p * mu
#     # return integrate.quad(lambda x: logPdf(x, l1, b1, l2, b2), 0, 100)[0] + math.log(p) + 1 / p * mu
#     #print(p)
#     # val = integrate.quad(lambda x: pdf(x, l1, l2, b1, b2), p, 1000)[0]
#     # #print(str(p) + " " + str(val) + ", mean: " + str(integrate.quad(lambda x: x*pdf(x, l1, l2, b1, b2), p, 1000)[0]))
#     # return val#integrate.quad(lambda x: pdf(x, l1, l2, b1, b2), p, 1000)[0]

def totalVariationDistance(l1,l2,b1,b2,p):
    return 0.5*integrate.quad(lambda x: abs(pdf(x,l1,l2,b1,b2)-math.exp(-x/p)/p), 0.0000001, 100)[0]

def hellingerDistance(l1,l2,b1,b2,p):
    return 1-integrate.quad(lambda x: math.sqrt(pdf(x,l1,l2,b1,b2)*math.exp(-x/p)/p), 0.0000001, 100)[0]

def approximateNoise(cov, N):
    D = np.diag(np.diag(cov))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    R = np.dot(np.dot(D_inv_sqrt, cov), D_inv_sqrt)
    mask = np.eye(R.shape[0], dtype=bool)
    # Shift the mask to exclude diagonal elements and their neighbors
    shifted_mask1 = np.roll(mask, shift=1, axis=1) | np.roll(mask, shift=-1, axis=1)
    shifted_mask2 = np.roll(mask, shift=2, axis=1) | np.roll(mask, shift=-2, axis=1)
    a1 = np.mean(R[shifted_mask1])
    a2 = np.mean(R[shifted_mask2])
    alpha = 0.5 * (a1 + np.sqrt(a2))
    p = ((1 - alpha ** 2) / (1 + alpha ** 2 - 2 * alpha * np.cos(2 * math.pi * np.arange(N) / N)))
    return p

def klDivergenceOverTime(mu, gamma, c, p):
    mu1 = np.real(mu)
    mu2 = np.imag(mu)
    a = np.diagonal(0.5 * np.real(gamma + c))
    d = np.diagonal(0.5 * np.real(gamma - c))
    b = np.diagonal(0.5 * np.imag(gamma + c))
    l1 = 0.5 * ((a + d) + np.sqrt((a - d) ** 2 + 4 * b ** 2))
    if np.any(l1<=0):
        loc = np.argwhere(l1<=0)
        print("Something weird")
        print(loc)
        print(l1[loc])
    l2 = 0.5 * ((a + d) - np.sqrt((a - d) ** 2 + 4 * b ** 2))
    p11 = 1 / np.sqrt((b ** 2 + (l1 - a) ** 2)) * b
    p21 = 1 / np.sqrt((b ** 2 + (l2 - a) ** 2)) * b
    p12 = 1 / np.sqrt((b ** 2 + (l1 - a) ** 2)) * (l1 - a)
    p22 = 1 / np.sqrt((b ** 2 + (l2 - a) ** 2)) * (l2 - a)
    if np.any(b < EPS):
        p11[b<EPS] = 1
        p12[b<EPS] = 0
        p21[b<EPS] = 0
        p22[b<EPS] = 1
    b1 = (mu1 * p11 + mu2 * p12) / np.sqrt(l1)
    b2 = (mu1 * p21 + mu2 * p22) / np.sqrt(l2)
    mu = mu1**2+mu2**2+a+d
    divergence = klDivergence(l1, l2, b1, b2, p, mu)
    # divergence = np.empty(len(l1))
    #for i, (ll1, ll2, bb1, bb2,mmu) in enumerate(zip(l1, l2, b1, b2,mu)):
    #    if i % 100 == 0:
    #        print(i)
        # divergence[i] = totalVariationDistance(ll1,ll2,bb1,bb2,p[i])#,mmu)
    #    divergence[i] = klDivergence(ll1,ll2,bb1,bb2,p[i],mmu)
    return divergence

def getPercentiles(l1, l2, b1, b2, min=0.00001, max=100, p=None):
  #if l2 < EPS:
    # Just consider first term
  #  return stats.ncx2.ppf(0.95, 1, nc)
  #print(str(l1) + " " + str(l2) + " " + str(b1) + " " + str(b2))
  if l2 < EPS:
      if l1 < EPS:
          cdfVal = lambda x: 0
      else:
          cdfVal = lambda x: cdfSingular(x, l=l1, b=b1, c=b2)
  else:
      # print("this one: " + str(l2))
      cdfVal = lambda x: cdf(x, l1=l1, l2=l2, b1=b1, b2=b2)
  #print("Min: " + str(cdfVal(min)) + " with " + str(min))
  #print("Max: " + str(cdfVal(max)) + " with " + str(max))
  #print("Testing")
  res = np.zeros((len(p)))
  # print("percentile")
  # print(min)
  # print(max)
  for i, percentile in enumerate(p):
      # print(percentile)
      # print(cdfVal(min))
      # print(cdfVal(max))
      try:
        res[i] = optimize.bisect(lambda x: cdfVal(x) - percentile, min, max)#, rtol=0.000001)
      except:
        print("Did not work for " + str(percentile))
        print(str(cdfVal(min)) + " " + str(cdfVal(max)))
  # median = optimize.bisect(lambda x: cdfVal(x) - 0.5, min, max, rtol=0.0001)
  # #print(cdfVal(median))
  # #print(str(median) + ", " + str(cdfVal(median)))
  # p95u = optimize.bisect(lambda x: cdfVal(x) - 0.025, min, max, rtol=0.0001)
  # p95o = optimize.bisect(lambda x: cdfVal(x) - 0.975, min, max, rtol=0.0001)
  # p50u = optimize.bisect(lambda x: cdfVal(x) - 0.25, min, max, rtol=0.0001)
  # p50o = optimize.bisect(lambda x: cdfVal(x) - 0.75, min, max, rtol=0.0001)
  # print(p95u)
  # print(p50u)
  #print(cdfVal(p95u))
  #print(cdfVal(p50u))
  # p95u = optimize.bisect(lambda x: cdfVal(x) - 0.025, min, median, rtol=0.001)
  # p95o = optimize.bisect(lambda x: cdfVal(x) - 0.975, median, max, rtol=0.001)
  # p50u = optimize.bisect(lambda x: cdfVal(x) - 0.25, p95u, median, rtol=0.001)
  # p50o = optimize.bisect(lambda x: cdfVal(x) - 0.75, median, p95o, rtol=0.001)
  return res#p95u, p50u, median, p50o, p95o

def percentilesOverTime(l1, l2, b1, b2, min=0.01, max=100, p = None):
    THRESHOLD = 0.01
    percentiles = np.zeros((len(p), len(l1)))
    # p95u_array = []
    # p50u_array = []
    # median_array = []
    # p50o_array = []
    # p95o_array = []
    #convMedian = []
    #convP95o = []

    counter = 0
    l1Max = np.max(l1)
    l2Max = np.max(l2)
    #b1Max = np.max(b1)
    #b2Max = np.max(b2)
    # print("l1Max " + str(l1Max))
    # print("l2Max " + str(l2Max))
    # print("b1Max " + str(b1Max))
    # print("b2Max " + str(b2Max))
    l1 = np.abs(l1)
    l2 = np.abs(l2)
    # Iterate over each set of parameters and compute the percentiles
    for i, (l1, l2, b1, b2) in enumerate(zip(l1, l2, b1, b2)):
        # if l1 < THRESHOLD * l1Max and l2 < THRESHOLD * l2Max:# or l1 < 10e-10 or l2 < 10e-10:
        if l1 < EPS and l2 < EPS:# or l1 < 10e-10 or l2 < 10e-10:
            # p95u_array.append(0)
            # p50u_array.append(0)
            # median_array.append(0)
            # p50o_array.append(0)
            # p95o_array.append(0)
            counter += 1
        # elif l1 < THRESHOLD*l1Max:
        # #
        # elif counter < 7:
        #     continue
        #else:
            #print("i=" + str(counter) + ": " + str(l1) + " " + str(l2) + " " + str(b1) + " " + str(b2))
            # xs = np.arange(num) * 1 - 0.2
            # pdfSamples = [pdf(x, l1=l1, l2=l2, b1=b1, b2=b2) for x in xs]
            # plt.plot(xs, pdfSamples)
            # plt.show()
            # print(min)
            # print(max)
            #p95u, p50u, median, p50o, p95o = getPercentiles(l1, l2, b1, b2, min=min, max=max)
        percentiles[:,i] = getPercentiles(l1, l2, b1, b2, min=min, max=max, p=p)
            # median = getPercentiles(l1, l2, b1, b2, min=min, max=max)
            # return 0,0,0,0,0
            # p95u_array.append(p95u)
            # p50u_array.append(p50u)
            # median_array.append(median)
            # p50o_array.append(p50o)
            # p95o_array.append(p95o)
            #convMedian.append(0.25/(pdf(median, l1, l2, b1, b2)**2))
            #convP95o.append(0.975*0.025/(pdf(p95o, l1, l2, b1, b2)**2))
        counter += 1

    # Convert the lists to arrays before returning if needed
    return percentiles#(np.array(p95u_array, dtype=float), np.array(p50u_array, dtype=float), np.array(median_array, dtype=float),
            #np.array(p50o_array, dtype=float), np.array(p95o_array, dtype=float))#, np.array(convMedian), np.array(convP95o))

def comutePercentiles(mu_X, Sigma, p=None):
    if p is None:
        p = [0.025, 0.25, 0.5, 0.75, 0.975]
    half = int(len(mu_X) / 2)
    a = Sigma.diagonal()[:half]
    d = Sigma.diagonal()[half:]
    b = Sigma[half:, :half].diagonal()
    #print("Det: " + str(a*d-b*b))
    # print("Test")
    # print(b[0])
    # print(Sigma[half, 0])
    # print(a[0])
    # print(Sigma[0,0])
    # print(d[0])
    # print(Sigma[half,half])
    # return 0,0,0,0,0
    # l1 = 0.5 * ((a + d) + np.sqrt((a - d) ** 2 + 4 * b ** 2))
    # l2 = 0.5 * ((a + d) - np.sqrt((a - d) ** 2 + 4 * b ** 2))
    return  computePercentilesDiagonals(mu_X[:half], mu_X[half:], a, b, d, p)

def computePercentilesComplex(mu, gamma, c, p):
    mu_re = np.real(mu)
    mu_im = np.imag(mu)
    a = np.diagonal(0.5 * np.real(gamma + c))
    d = np.diagonal(0.5 * np.real(gamma - c))
    b = np.diagonal(0.5 * np.imag(gamma + c))
    return computePercentilesDiagonals(mu_re, mu_im, a, b, d, p)

def computePercentilesDiagonals(mu1, mu2, a, b, d, p=None):
    if p is None:
        p = [0.025, 0.25, 0.5, 0.75, 0.975]
    l1 = 0.5 * ((a + d) + np.sqrt((a - d) ** 2 + 4 * b ** 2))
    l2 = 0.5 * ((a + d) - np.sqrt((a - d) ** 2 + 4 * b ** 2))
    if np.any(l2 < EPS):
        d = np.copy(d)
        d[d<0] = 0
        a = np.copy(a)
        a[a<0] = 0
        b1 = np.zeros(l1.shape)
        b2 = np.zeros(l1.shape)
        b1[l2<EPS] = mu1[l2 < EPS]*np.sqrt(a[l2 < EPS])+mu2[l2 < EPS]*np.sqrt(d[l2 < EPS])
        if np.any(l2 >= EPS):
            p11 = 1 / np.sqrt((b[l2 >= EPS] ** 2 + (l1[l2 >= EPS] - a[l2 >= EPS]) ** 2)) * b[l2 >= EPS]
            p21 = 1 / np.sqrt((b[l2 >= EPS] ** 2 + (l2[l2 >= EPS] - a[l2 >= EPS]) ** 2)) * b[l2 >= EPS]
            p12 = 1 / np.sqrt((b[l2 >= EPS] ** 2 + (l1[l2 >= EPS] - a[l2 >= EPS]) ** 2)) * (l1[l2 >= EPS] - a[l2 >= EPS])
            p22 = 1 / np.sqrt((b[l2 >= EPS] ** 2 + (l2[l2 >= EPS] - a[l2 >= EPS]) ** 2)) * (l2[l2 >= EPS] - a[l2 >= EPS])
            b2[l2 >= EPS] = (mu1[l2 >= EPS] * p21 + mu2[l2 >= EPS] * p22) / np.sqrt(l2[l2 >= EPS])
            b1[l2 >= EPS] = (mu1[l2 >= EPS] * p11 + mu2[l2 >= EPS] * p12) / np.sqrt(l1[l2 >= EPS])
        if np.any(a + d < EPS):
            l1[a + d < EPS] = 0
        mask = np.logical_and(l2<EPS, a+d>=EPS)
        b2[mask] = mu1[mask]**2+mu2[mask]**2-(b1[mask])**2/(a[mask]+d[mask])
        #b1[l2>=10e-7] =  (mu_X[1:half] * p21 + mu_X[half + 1:] * p22)[l2>=10e-7] / np.sqrt(l2[l2>=10e-7])
    else:
        p11 = 1 / np.sqrt((b ** 2 + (l1 - a) ** 2)) * b
        p21 = 1 / np.sqrt((b ** 2 + (l2 - a) ** 2)) * b
        p12 = 1 / np.sqrt((b ** 2 + (l1 - a) ** 2)) * (l1 - a)
        p22 = 1 / np.sqrt((b ** 2 + (l2 - a) ** 2)) * (l2 - a)
        if np.any(np.abs(b) < EPS):
            # print("Test")
            # print(b[np.abs(b) < EPS])
            # print(b1[np.abs(b) < EPS])
            # print(b2[np.abs(b) < EPS])
            # print(l1[np.abs(b) < EPS])
            # print(l2[np.abs(b) < EPS])
            p11[b < EPS] = 1
            p12[b < EPS] = 0
            p21[b < EPS] = 0
            p22[b < EPS] = 1
        b1 = (mu1 * p11 + mu2 * p12) / np.sqrt(l1)
        b2 = (mu1 * p21 + mu2 * p22) / np.sqrt(l2)
    #print("Compute percentiles")
    # return l1, l2, b1, b2, b2#
    return percentilesOverTime(l1, l2, b1, b2, min=0, max=max(1000, np.max((mu1**2+mu2**2)*10)), p=p)

# Visualization
def mds(d, dimensions=3):
    """
    Multidimensional Scaling - Given a matrix of interpoint distances,
    find a set of low dimensional points that have similar interpoint
    distances.
    """

    (n, n) = d.shape
    E = (-0.5 * d ** 2)

    # Use mat to get column and row means to act as column and row means.
    Er = np.mat(np.mean(E, 1))
    Es = np.mat(np.mean(E, 0))

    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = np.array(E - np.transpose(Er) - Es + np.mean(E))

    [U, S, V] = np.linalg.svd(F)

    Y = U * np.sqrt(S)

    return (Y[:, 0:dimensions], S)

def getColors(pos):
    # Treat x and y component as a*,b* component, scale to optimize space
    # a* in [-128, 127], b* in [-128, 127]
    # Scale to 0, 1, then to desired range
    L = 50
    abSpace = (pos-np.min(pos))/(np.max(pos)-np.min(pos))*255-128
    LabSpace = np.ones((len(abSpace), 3))*L
    LabSpace[:,1:] = abSpace
    # Convert CIELab to RGB
    colors = color.lab2rgb(LabSpace)
    return colors

def bestpair(points):
    hull = ConvexHull(points)
    hullpoints = points[hull.vertices,:]
    # Naive way of finding the best pair in O(H^2) time if H is number of points on
    # hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    return hullpoints[bestpair[0]], hullpoints[bestpair[1]]

def rotate_points(points):
    p1, p2 = bestpair(points)
    # Get the angle between the points and the diagonal
    vec = p2-p1
    vec_norm = vec/np.linalg.norm(vec)
    # Rotate around vector perpendicular to the other two
    direction = np.array([1,1,1])/np.sqrt(3)
    n = np.cross(vec_norm, direction)
    n = n/np.linalg.norm(n)
    rot_angle = math.acos(np.dot(vec_norm, direction))
    cosA = math.cos(rot_angle)
    sinA = math.sin(rot_angle)
    rot_mat = np.array([[n[0]**2*(1-cosA)+cosA, n[0]*n[1]*(1-cosA)-n[2]*sinA, n[0]*n[2]*(1-cosA)+n[1]*sinA],
                        [n[1]*n[0]*(1-cosA)+n[2]*sinA, n[1]**2*(1-cosA)+cosA, n[1]*n[2]*(1-cosA)-n[0]*sinA],
                        [n[2]*n[0]*(1-cosA)-n[1]*sinA, n[2]*n[1]*(1-cosA)+n[0]*sinA, n[2]**2*(1-cosA)+cosA]])
    # Rotate points:
    points_rot = np.empty(points.shape)
    for i in range(len(points)):
        points_rot[i] = np.matmul(rot_mat, points[i])
    return points_rot

# Normalize with the same scaling factor in all directions
def normalize_point(points):
    points_min_vector = np.min(points)
    points_max_vector = np.max(points)
    points_normalized = (points - points_min_vector) / (points_max_vector - points_min_vector)
    return points_normalized
def getColors3D(pos):
    # Treat x and y component as a*,b* component, scale to optimize space
    # a* in [-128, 127], b* in [-128, 127]
    # Scale to 0, 1, then to desired range
    mds_points = rotate_points(pos)
    mds_points = normalize_point(pos)
    pos[:,0]=pos[:,0]*100
    pos[:,1]=pos[:,1]*100-50
    pos[:,2]=pos[:,2]*100-50
    colors = color.lab2rgb(pos)
    return colors

def colorCodeBand(mean, cov, colors):
    # https://stackoverflow.com/questions/68002782/fill-between-gradient
    variance = np.sqrt(cov.diagonal())
    polygon = plt.fill_between(np.arange(len(mean)), mean-variance, mean+variance, lw=0, color='none')
    xlim = plt.xlim()
    ylim = plt.ylim()
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    #print(colors.shape)
    cmap = ListedColormap(colors)
    gradient = plt.imshow(np.linspace(0, 1, len(mean)).reshape(1, -1), cmap=cmap, aspect='auto',
                          extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.show()

def plotCovarianceAsColors(mean, cov, xmin = 0, xmax = 0):
    if xmax == 0:
        xmax = len(mean)
    distanceCov = 1 - (cov - np.min(cov)) / (np.max(cov) - np.min(cov))
    positions, eigenvalues = mds(distanceCov, dimensions=2)
    #print("Eigenvalues: " + str(eigenvalues[:4]))
    #print("Percentages of variation covered by eigenvalues: " + str(eigenvalues[:4]/np.sum(eigenvalues)))
    colors = getColors(positions)
    colorCodeBand(mean[xmin:xmax], cov[xmin:xmax, xmin:xmax], colors[xmin:xmax])