import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import sys
sys.path.append('../..')
import uafourier

np.random.seed(0)

plt.rcParams['font.size'] = 7

covDensity = np.load("covDensity.npy")[:250]
muDensity = np.load("muDensity.npy")#[:250]
percentiles = np.load("percentiles.npy")[:,:250]
fftMuTest = np.load("fftMuTest.npy")
fftGammaTest = np.load("fftGammaTest.npy")
fftCTest = np.load("fftCTest.npy")
f = (np.arange(len(percentiles[0]))) / len(percentiles[0]) / (2*1/12)
f = f[:250]

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10/2.54, 11/2.54))  # Convert size from cm to inches

std_dev = np.sqrt(np.diag(covDensity))
mean_line, = axs[0].plot(f, muDensity, label='Expected Value', color='blue')
std1 = axs[0].fill_between(f, muDensity  - std_dev, muDensity + std_dev, alpha=0.3, label=r'Standard deviation', color='blue', edgecolor='none')
axs[0].set_title("Using mean and covariance")

axs[0].set_xlim(0,0.75)
axs[0].set_ylim(-5,80)

axs[0].set_xlabel(r'$f$')
axs[0].set_ylabel(r'Energy density')
axs[0].legend(frameon=False, loc='upper left')

std_dev = np.sqrt(np.diag(covDensity))
mean_line, = axs[1].plot(f, percentiles[2], label='Expected Value', color='blue')
std1 = axs[1].fill_between(f, percentiles[1], percentiles[3], alpha=0.3, label=r'50$\%$', color='blue', edgecolor='none')
std1 = axs[1].fill_between(f, percentiles[0], percentiles[1], alpha=0.1, label=r'95$\%$', color='blue', edgecolor='none')
std1 = axs[1].fill_between(f, percentiles[3], percentiles[4], alpha=0.1, label='_nolegend_', color='blue', edgecolor='none')
axs[1].set_title("Using percentiles")

# Insert subplot (https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib)
rect=[0.82,0.4,0.25,0.5]
figS = plt.gcf()
box = axs[1].get_position()
width = box.width
height = box.height
inax_position  = axs[1].transAxes.transform(rect[0:2])
transFigure = figS.transFigure.inverted()
infig_position = transFigure.transform(inax_position)
x = infig_position[0]
y = infig_position[1]
width *= rect[2]
height *= rect[3]
subax = figS.add_axes([x,y,width,height],facecolor='w')
x_labelsize = subax.get_xticklabels()[0].get_size()
y_labelsize = subax.get_yticklabels()[0].get_size()
xs = np.linspace(0, 10, 100)
position = 23#17
a = 0.5*np.real(fftGammaTest+ fftCTest)[position, position]
d = 0.5*np.real(fftGammaTest- fftCTest)[position, position]
b = 0.5*np.imag(fftGammaTest+ fftCTest)[position, position]
l1 = 0.5 * ((a + d) + np.sqrt((a - d) ** 2 + 4 * b ** 2))
l2 = 0.5 * ((a + d) - np.sqrt((a - d) ** 2 + 4 * b ** 2))
p11 = 1 / np.sqrt((b ** 2 + (l1 - a) ** 2)) * b
p21 = 1 / np.sqrt((b ** 2 + (l2 - a) ** 2)) * b
p12 = 1 / np.sqrt((b ** 2 + (l1 - a) ** 2)) * (l1 - a)
p22 = 1 / np.sqrt((b ** 2 + (l2 - a) ** 2)) * (l2 - a)
mu1 = np.real(fftMuTest[position])
mu2 = np.imag(fftMuTest[position])
b1 = (mu1 * p11 + mu2 * p12) / np.sqrt(l1)
b2 = (mu1 * p21 + mu2 * p22) / np.sqrt(l2)
pdf = [uafourier.pdf(x, l1=l1, l2=l2, b1=b1, b2=b2) for x in xs]
subax.set_xlabel("Energy density", labelpad=-1)
subax.set_ylabel("PDF", labelpad=-1)
subax.plot(xs, pdf)
print(f[position])

axs[1].set_xlabel(r'$f$')
axs[1].set_ylabel(r'Energy density')
axs[1].legend(frameon=False, loc='upper left')


axs[1].set_xlim(0,0.75)
axs[1].set_ylim(-5,80)

ellipse = Ellipse(xy=(f[position], muDensity[position]+3), width=0.03, height=15, 
                        edgecolor=(0,100/255,0,1), fc='None', lw=2)
axs[1].add_patch(ellipse)

l = Line2D([f[position], 0.57], [muDensity[position]+10.5, 40], color=(0,100/255,0,1), lw=2)
axs[1].add_line(l)

plt.tight_layout()
plt.savefig("fourier.pdf", bbox_inches='tight')
plt.show()