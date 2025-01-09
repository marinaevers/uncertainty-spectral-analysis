import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
import os
import netCDF4 as nc
import sys
sys.path.append('../..')
import uafourier
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats

# Figures for revision that showcase the results if the transformation is directly applied to samples

def loadMPIGE():
    ENSEMBLE_PATH = r"T:\tas\rcp85"
    ensembleRes = 100
    temp = np.zeros((ensembleRes, 1128))
    xmin = 190  # 120
    xmax = 240  # 170
    run = os.path.join(ENSEMBLE_PATH, sorted(os.listdir(ENSEMBLE_PATH))[0])
    data = nc.Dataset(run, "r")
    lonFilter = np.logical_and(np.array(data['lon']) > xmin, np.array(data['lon']) < xmax)
    latFilter = np.abs(data['lat']) < 5
    # Create a meshgrid of the filters to match the dimensions of temp
    lonFilter_mesh, latFilter_mesh = np.meshgrid(lonFilter, latFilter)

    # Combine the filters
    combinedFilter = np.logical_and(lonFilter_mesh, latFilter_mesh)

    for i, f in enumerate(sorted(os.listdir(ENSEMBLE_PATH))):
        run = os.path.join(ENSEMBLE_PATH, f)
        data = nc.Dataset(run, "r")
        temp[i] = np.mean(np.array(data["tas"])[:, combinedFilter], axis=1)
        monthlyMeans = np.mean(temp[i][:10 * 12].reshape((10, 12)), axis=0)
        temp[i] = temp[i] - np.tile(monthlyMeans, int(len(temp[0]) / 12))
        Xx = [i for i in np.arange(0, len(temp[i]))]
        Xx = np.reshape(Xx, (len(Xx), 1))
        yy = temp[i]
        model = sklearn.linear_model.LinearRegression()
        model.fit(Xx, yy)
        # calculate trend
        trend = model.predict(Xx)
        temp[i] = [yy[i] - trend[i] for i in np.arange(0, len(temp[i]))]
    return temp

def loadRossmann():
    data = pd.read_csv("T:\\rossmann-store-sales\\train.csv", low_memory=False)
    data['Date'] = pd.to_datetime(data['Date'])
    # Create numpy time series data
    store_sales = []
    count_invalid = 0
    for store_id, group in data.groupby("Store"):
        group = group.sort_values('Date')
        sales_array = group['Sales'].values
        if (len(sales_array) != 942):
            count_invalid += 1
            continue
        # Normalize the data with respect to their means
        sales_array = sales_array.astype(float) / np.mean(sales_array.astype(float))
        # plt.plot(sales_array)
        store_sales.append(sales_array)
    store_sales = np.array(store_sales)
    return store_sales

def getPercentilesNormal(data):
    mean = np.mean(data, axis=0)
    sigma = np.cov(data.T)
    lower95 = mean - 1.96 * np.sqrt(sigma.diagonal())
    lower50 = mean - 0.6745 * np.sqrt(sigma.diagonal())
    upper50 = mean + 0.6745 * np.sqrt(sigma.diagonal())
    upper95 = mean + 1.96 * np.sqrt(sigma.diagonal())
    return lower95, lower50, mean, upper50, upper95

def getPercentilesSamples(data, remove0 = False):
    median = np.median(data, axis=0)
    lower95 = np.percentile(data, 2.5, axis=0)
    lower50 = np.percentile(data, 25, axis=0)
    upper50 = np.percentile(data, 75, axis=0)
    upper95 = np.percentile(data, 97.5, axis=0)
    # remove frequency 0 because it is not really of interest and makes everything else
    # harder to interpret
    if remove0:
        lower95[0] = 0
        lower50[0] = 0
        median[0] = 0
        upper50[0] = 0
        upper95[0] = 0
    return lower95, lower50, median, upper50, upper95

def transformSamples(data):
    samples_f = []
    N = len(data[0])
    W = 1 / np.sqrt(N) * np.array([[uafourier.fourier(i, j, N) for j in range(N)] for i in range(N)], dtype=complex)
    for i in range(len(data)):
        if not samples_f:
            samples_f = [np.dot(W, data[i])]
        else:
            samples_f.append(np.dot(W, data[i]))
    samples_f = np.array(samples_f)
    samples_s = np.abs(samples_f) ** 2
    return samples_s

def getPercentilesTransformed(data):
    mean = np.mean(data, axis=0)
    sigma = np.cov(data.T)
    fftMu, fftGamma, fftC = uafourier.ua_fourier(mean, sigma)
    mu_f, sigma_f = uafourier.complex_to_real(fftMu, fftGamma, fftC)
    p95u, p50u, median, p50o, p95o = uafourier.comutePercentiles(mu_f, sigma_f)
    return p95u, p50u, median, p50o, p95o

def plot(l95n, l50n, meanNormal, u50n, u95n, xMax, l95s, l50s, medianSamples, u50s, u95s, path, xlabel="Time Step", ylabel="Value", x = None):
    xx = x
    plt.plot(xx, meanNormal, label="Mean")
    # 50% band
    plt.fill_between(xx, l50n, u50n, alpha=0.2, color='blue', label="50%")
    # 95% band
    plt.fill_between(xx, l95n, u95n, alpha=0.1, color='blue', label="95%")

    plt.plot(xx, l95s, '--', c='#d95f02', label="Samples")
    plt.plot(xx, l50s, '--', c='#d95f02')
    plt.plot(xx, medianSamples, '--', c='#d95f02')
    plt.plot(xx, u50s, '--', c='#d95f02')
    plt.plot(xx, u95s, '--', c='#d95f02')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xlim(x[0], x[xMax])
    yMax = int(max(np.max(u95s[:xMax]), np.max(u95n[:xMax])))+1
    yMin = int(min(np.min(l95s[:xMax])*1.1, np.min(l95n[:xMax]))*1.1)-0.5
    yMin = min(-0.1, yMin)
    plt.ylim((yMin, yMax))
    plt.savefig(path)
    plt.show()

def pltHist(data, p, xlabel, ylabel, filename = ""):
    plt.figure(figsize=(2,2))
    plt.hist(data, density = True, bins=20)
    mean = np.mean(data)
    std = np.std(data)
    xx = np.linspace(np.min(data), np.max(data), 100)
    pdf = [stats.norm.pdf(x, mean, std) for x in xx]
    plt.plot(xx, pdf)
    plt.title("p = " + str(p))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

data = loadMPIGE()
l95n, l50n, meanNormal, u50n, u95n = getPercentilesNormal(data)
l95s, l50s, medianSamples, u50s, u95s = getPercentilesSamples(data)
start_date = datetime(2006, 1, 1)
dates = [start_date + timedelta(days=30*i) for i in range(len(meanNormal))]
plot(l95n, l50n, meanNormal, u50n, u95n, 100, l95s, l50s, medianSamples, u50s, u95s, "mpigeInput.pdf", xlabel="", ylabel="Temperature", x=dates)
spectralSamples = transformSamples(data)
l95s, l50s, medianSamples, u50s, u95s = getPercentilesSamples(spectralSamples[:,:int(len(spectralSamples[0])/2)], True)
l95t, l50t, medianTransformed, u50t, u95t = getPercentilesTransformed(data)
f = np.arange(len(medianSamples))/len(medianSamples)*6
plot(l95t, l50t, medianTransformed, u50t, u95t, 100, l95s, l50s, medianSamples, u50s, u95s, "mpigeOutput.pdf", xlabel="Frequency", ylabel="Energy Density", x=f)
data = data.T
p = np.zeros(len(data))
print(data.shape)
for t in range(len(data)):
    _, p[t] = stats.shapiro(data[t])
print(p.shape)
maxP = np.argmax(p)
minP = np.argmin(p)
medianP = np.argsort(p)[len(p)//2]
print(np.count_nonzero(p < 0.05))
pltHist(data[maxP], p[maxP], "Temp. Anomaly", "Probability", "mpigePMax.pdf")
pltHist(data[minP], p[minP], "Temp. Anomaly", "Probability", "mpigePMin.pdf")
pltHist(data[medianP], p[medianP], "Temp. Anomaly", "Probability", "mpigePMedian.pdf")

data = loadRossmann()
l95n, l50n, meanNormal, u50n, u95n = getPercentilesNormal(data)
l95s, l50s, medianSamples, u50s, u95s = getPercentilesSamples(data)
start_date = datetime(2013, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(len(meanNormal))]
plot(l95n, l50n, meanNormal, u50n, u95n, 100, l95s, l50s, medianSamples, u50s, u95s, "rossmannInput.pdf",xlabel="", ylabel="Sales", x=dates)
spectralSamples = transformSamples(data)
l95s, l50s, medianSamples, u50s, u95s = getPercentilesSamples(spectralSamples[:,:int(len(spectralSamples[0])/2)], True)
l95t, l50t, medianTransformed, u50t, u95t = getPercentilesTransformed(data)
f = np.arange(len(medianSamples))/len(medianSamples)*0.5
plot(l95t, l50t, medianTransformed, u50t, u95t, 100, l95s, l50s, medianSamples, u50s, u95s, "rossmannOutput.pdf", xlabel="Frequency", ylabel="Energy Density", x=f)
data = data.T
p = np.zeros(len(data))
print(data.shape)
for t in range(len(data)):
    _, p[t] = stats.shapiro(data[t])
print(p.shape)
maxP = np.argmax(p)
medianP = np.argsort(p)[len(p)//2]
p[p==0] = 2
print(np.count_nonzero(p==2))
minP = np.argmin(p)
print(np.count_nonzero(p < 0.05))
pltHist(data[maxP], p[maxP], "Sales", "Probability", "rossmannPMax.pdf")
pltHist(data[minP], p[minP], "Sales", "Probability", "rossmannPMin.pdf")
pltHist(data[medianP], p[medianP], "Sales", "Probability", "rossmannPMedian.pdf")