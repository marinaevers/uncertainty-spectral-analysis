import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

plt.rcParams['font.size'] = 7

x = np.arange(500)*0.02
f = np.sin(x) + 0.01*x**2
#plt.plot(x, f)

# Function definition
def f(x):
    return np.sin(x) + 0.014 * x**2

# Create x values
x = np.arange(500) * 0.03

# Compute the mean
mean = f(x)

cov1 = np.eye(len(x)) * 0.1

# Create a valid covariance matrix with exponential decay
decay_factor = 0.1
cov2 = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        cov2[i, j] = np.exp(-decay_factor * np.abs(x[i] - x[j])) * 0.1

# Generate three samples from each covariance matrix
num_samples = 1
samples1 = np.random.multivariate_normal(mean, cov1, num_samples)
samples2 = np.random.multivariate_normal(mean, cov2, num_samples)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10/2.54, 11/2.54))  # Convert size from cm to inches

for i, (samples, cov, ax) in enumerate(zip([samples1, samples2], [cov1, cov2], axs)):
    # Compute standard deviation and 3 times standard deviation
    std_dev = np.sqrt(np.diag(cov))
    std_3 = 1.96 * std_dev

    # Plot mean, standard deviation bands and samples
    mean_line, = ax.plot(x, mean, label='Expected Value', color='blue')
    std1 = ax.fill_between(x, mean  - 0.6745 * std_dev, mean + 0.6745 * std_dev, alpha=0.3, label=r'50$\%$', color='blue', edgecolor='none')
    std3 = ax.fill_between(x, mean + 0.6745 * std_dev, mean + std_3, alpha=0.1, label=r'95$\%', color='blue', edgecolor='none')
    std3 = ax.fill_between(x, mean - std_3, mean - 0.6745 * std_dev, alpha=0.1, label=r'3$\sigma$', color='blue',edgecolor='none')

    s = None
    for sample in samples:
        s, = ax.plot(x, sample, c='gray', alpha=0.8, label='Sample')#, linestyle='--')

    if i == 0:
        ax.set_title("No cross-covariance")
    else:
        ax.set_title("Covariance")
    ax.set_xlim(0,15)
    ax.set_ylim(-2,5)
    
    handles = [mean_line, std1, std3, s]
    labels = ['Expected value', r'50$\%$ interval', r'95$\%$ interval', 'Sample']
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.legend(handles, labels, frameon=False)

plt.tight_layout()
plt.savefig("covariance.pdf", bbox_inches='tight')
plt.show()