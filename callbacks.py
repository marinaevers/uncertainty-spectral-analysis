import base64
import io
import json

from dash import callback, Output, Input, State
import plotly.graph_objects as go
import numpy as np
from dash.exceptions import PreventUpdate
import uafourier
import math
from joblib import Memory
from PIL import Image
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd

import matplotlib.pyplot as plt

memory = Memory("cachedir")

options = ['Expected Value', '50% Range', '95% Range', '2.5th Percentile', '25th Percentile', 'Median', '75th Percentile', '97.5th Percentile']

TEMPLATE = 'simple_white'

@callback(
    Output('store-data', 'data'),
    Output('threshold-slider', 'max'),
    Output('slider1', 'max'),
    Output('slider2', 'max'),
    Input('upload-data-mean', 'contents'),
    Input('upload-data-cov', 'contents'),
    Input('upload-data-config', 'contents'),
    Input('switch-distance-noise','on'),
    State('store-data', 'data')
    #State('slider-timesteps-max', 'value')
)
def update_data(contentsMean, contentsCov, contentsConfig, distanceToNoise, store):#, timesteps):
    if not store:
        store = {}

    maxTime = 1
    maxCov = 1
    def load_numpy_array(base64content):
        _, content_string = base64content.split(',')
        decoded = base64.b64decode(content_string)
        buffer = io.BytesIO(decoded)
        return np.load(buffer, allow_pickle=True)

    if contentsConfig:
        _, contents = contentsConfig.split(',')
        contents = base64.b64decode(contents)
        config = json.loads(contents)
        store['dt'] = config['dt']
        store['dj'] = config['dj']
        store['zres'] = config['zres']
        store['showDate'] = config['date']
        store['startDate'] = config['startDate']
        store['offset'] = config['offset']
        store['freq'] = 'ME'
        if 'freq' in config:
            store['freq'] = config['freq']

    if contentsMean:
        _, contents = contentsMean.split(',')
        mean = load_numpy_array(contentsMean)
        store['mean-temporal'] = mean#[:timesteps]
        maxTime = len(mean)

    if contentsCov:
        _, contents = contentsCov.split(',')
        # mean = np.frombuffer(base64.b64decode(contents))
        cov = load_numpy_array(contentsCov)
        store['cov-temporal'] = cov#[:timesteps, :timesteps]
        maxCov = np.max(cov.diagonal())

    # Compute transformation if data available
    if 'mean-temporal' in store and 'cov-temporal' in store:
        mu = store['mean-temporal']
        sigma_t = store['cov-temporal']

        fftMu, fftGamma, fftC = uafourier.ua_fourier(mu, sigma_t)
        store['fftMean-real'] = np.real(fftMu)
        store['fftMean-im'] = np.imag(fftMu)
        store['fftGamma-real'] = np.real(fftGamma)
        store['fftGamma-imag'] = np.imag(fftGamma)
        store['fftC-real'] = np.real(fftC)
        store['fftC-imag'] = np.imag(fftC)

        dj = store['dj']
        dt = store['dt']
        maxTime = len(mu)*dt
        s_0 = 2 * dt
        J = int(math.log2(len(fftMu) * dt / s_0) / dj)

        @memory.cache
        def getWavelet(fftMu, fftGamma, fftC, dj, dt, s_0, J, p):
            expectation = np.empty((J, len(fftMu))) 
            complexExpectation = np.empty((J, len(fftMu)),dtype='complex')
            var = np.empty((J, len(fftMu)))
            var_re_re = np.empty((J, len(fftMu)))
            var_im_im = np.empty((J, len(fftMu)))
            var_re_im = np.empty((J, len(fftMu)))
            scales = np.zeros(J)
            percentiles = np.empty((5, J, len(fftMu)))
            distance = np.empty((J, len(fftMu)))
            w_0 =6
            beta = 4*np.pi/(w_0+np.sqrt(2+w_0**2))
            # Determine next power of two
            N = len(fftMu)
            Nn = N#(1<<(N-1).bit_length())
            start = 0#int((Nn-N)/2)
            end = start + N
            # paddedMu = np.zeros(Nn, dtype='complex')
            # paddedMu[start:end] = fftMu
            # paddedGamma = np.zeros((Nn, Nn), dtype='complex')
            # paddedGamma[start:end,start:end] = fftGamma
            # paddedC = np.zeros((Nn, Nn), dtype='complex')
            # paddedC[start:end,start:end] = fftC
            start = int((Nn-N)/2)
            end = start + N
            for i in range(J):
                print(i)
                waveletMu, waveletGamma, waveletC = uafourier.wavelet(fftMu, fftGamma, fftC, s_0 * math.pow(2, i * dj), dt)
                # waveletMu, waveletGamma, waveletC = uafourier.wavelet(paddedMu, paddedGamma, paddedC, s_0 * math.pow(2, i * dj), dt)
                waveletMu = waveletMu[start:end]
                waveletC = waveletC[start:end,start:end]
                waveletGamma = waveletGamma[start:end,start:end]
                var_re_re[i] = np.diag(0.5 * np.real(waveletGamma + waveletC))
                var_im_im[i] = np.diag(0.5 * np.real(waveletGamma - waveletC))
                var_re_im[i] = np.diag(0.5 * np.imag(waveletGamma + waveletC))
                # expectation[i] = np.real(waveletMu) ** 2 + np.imag(waveletMu) ** 2 + np.diag(
                #     0.5 * np.real(waveletGamma + waveletC)) + np.diag(0.5 * np.real(waveletGamma - waveletC))
                expectation[i] = np.real(waveletMu) ** 2 + np.imag(waveletMu) ** 2 + var_re_re[i] + var_im_im[i]
                complexExpectation[i] = waveletMu
                var_re_re_square = 2*var_re_re[i]**2 + 4*np.real(waveletMu)**2*var_re_re[i]
                var_im_im_square = 2*var_im_im[i]**2 + 4*np.imag(waveletMu)**2*var_im_im[i]
                var_re_im_square = 2*var_re_im[i]**2 + 4*np.real(waveletMu)*np.imag(waveletMu) * var_re_im[i]
                var_im_re_square = var_re_im_square.T
                var[i] = var_re_re_square + var_im_im_square + var_re_im_square + var_im_re_square
                scales[i] = s_0 * math.pow(2, i * dj)
                percentiles[:,i] = uafourier.computePercentilesComplex(waveletMu, waveletGamma, waveletC, [0.05, 0.25, 0.5, 0.75, 0.95])
                k = int(round(1 / (beta * scales[i]) * len(fftMu) * dt))
                distance[i] = uafourier.klDivergenceOverTime(waveletMu, waveletGamma, waveletC, p[k]*5.99/2)
            return expectation, scales, percentiles, distance, var, complexExpectation, var_re_re, var_im_im, var_re_im
        sigma2 = np.sqrt(np.mean(np.diagonal(sigma_t)))
        p = uafourier.approximateNoise(sigma_t, len(mu))
        store['noise-spectrum'] = p
        expectation, scales, percentiles, distance, var, complexExpectation, var_re_re, var_im_im, var_re_im = getWavelet(fftMu/sigma2, fftGamma/sigma2**2, fftC/sigma2**2,dj, dt, s_0, J, p)
        store['wavelet-expectation'] = expectation
        store['wavelet-expectation-real'] = np.real(complexExpectation)
        store['wavelet-expectation-imag'] = np.imag(complexExpectation)
        store['wavelet-var'] = var
        store['wavelet-scales'] = scales
        store['wavelet-percentiles'] = percentiles
        np.save("percentiles.npy", percentiles)
        store['wavelet-var-re-re'] = var_re_re
        store['wavelet-var-re-im'] = var_re_im
        store['wavelet-var-im-im'] = var_im_im
        distance = np.nan_to_num(distance)
        store['wavelet-distance'] = distance
        N = expectation.shape[1]
        timescale = np.arange(N) * store['dt']
        s_values = 1 / np.sqrt(2) * timescale
        s_values[int(N / 2):] = 1 / np.sqrt(2) * timescale[:int(N / 2)][::-1]
        s_values[s_values < 2 * store['dt']] = 2 * store['dt']
        s_values = 1 / store['dj'] * np.log2(s_values / (2 * store['dt']))
        store['sValues'] = s_values
        coiFiltering = np.ones(expectation.shape)
        x = np.arange(len(scales))
        for i, s_value in enumerate(s_values):
            coiFiltering[x < s_value, i] = 0
        store['coiFiltering'] = coiFiltering
        # Compute percentiles for Fourier
        half = int(len(fftMu)/2)
        percentiles = uafourier.computePercentilesComplex(fftMu[:half], fftGamma[:half, :half], fftC[:half, :half], [0.05, 0.25, 0.5, 0.75, 0.95])
        store['percentiles-fourier'] = percentiles
    upper_bound = 0
    if 'wavelet-distance' in store and distanceToNoise:
        upper_bound = np.max(store['wavelet-distance'])
    elif 'wavelet-expectation' in store and not distanceToNoise:
        upper_bound = np.max(store['wavelet-expectation'])
    return store, upper_bound, maxTime, maxCov

@callback(
    Output('label-slider-1', 'children'),
    Output('label-slider-2', 'style'),
    Output('slider1-div', 'style'),
    Output('sliderVal-div', 'style'),
    Output('slider2-div', 'style'),
    Output('slider3-div', 'style'),
    Input('dropdown-distribution-selection', 'value')
)
def update_distribution_change(value):
    if value == "normal":
        return "Mean", {"display":"block"}, {"display":"block"}, {"display":"none"}, {"display":"block"}, {"display":"block"}
    elif value == "uniform":
        return "Value", {"display":"none"}, {"display":"none"}, {"display":"block"}, {"display":"none"}, {"display":"none"}
    raise PreventUpdate


@callback(
    Output('fig-time-series', 'figure'),
    Input('store-data', 'data'),
    Input('store-perturbation', 'data'),
    Input('tabs', 'value')
)
def update_fig_temporal(data, dataPert, tabValue):
    # Show mean, 50% quantile and 95% quantile
    if not data or not 'mean-temporal' in data or not 'cov-temporal' in data:
        raise PreventUpdate

    std_dev = np.sqrt(np.diagonal(data['cov-temporal']))
    mean = np.array(data['mean-temporal'])+data['offset']


    if data['showDate']:
        x = pd.date_range(start=data['startDate'], periods=len(mean), freq=data['freq'])
    else:
        x = np.arange(len(mean))*data['dt']

    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=mean, mode='lines', name='Mean'))

    # Plotting the 50% percentile lines (mean ± 0.6745*standard deviation)
    percentile50lower = mean - 0.6745 * std_dev
    percentile50upper = mean + 0.6745 * std_dev
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([percentile50lower, percentile50upper[::-1]]),
                             fill='toself',
                             fillcolor='rgba(0, 0, 255, 0.3)',  # Blue with low opacity
                             line=dict(color='rgba(255, 255, 255, 0)'), name='50%'))

    # Plotting the 95% percentile lines (mean ± 1.96*standard deviation)
    percentile95lower = mean - 1.96 * std_dev
    percentile95upper = mean + 1.96 * std_dev
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([percentile95lower, percentile95upper[::-1]]),
                             fill='toself',
                             fillcolor='rgba(0, 0, 255, 0.1)',  # Blue with lower opacity
                             line=dict(color='rgba(255, 255, 255, 0)'), name='95%'))

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Value',
        margin=dict(l=40, r=0, t=25, b=0),
        title='Time Series',
        template=TEMPLATE
    )
    if data['showDate']:
        fig.update_layout(
            xaxis=dict(
                type='date'
            )
        )

    if tabValue == 'sensitivity' and dataPert:
        std_dev = np.sqrt(np.diagonal(data['cov-temporal'])+np.diagonal(dataPert['theta']))
        p50Lower_p = mean - 0.6745 * std_dev
        p50Upper_p = mean + 0.6745 * std_dev
        p95Lower_p = mean - 1.96 * std_dev
        p95Upper_p = mean + 1.96 * std_dev
        fig.add_trace(go.Scatter(x=x, y=p95Lower_p, mode='lines',
                                 name='Percentiles (perturbation)',
                                 line=dict(dash='dot', color='rgba(0, 0, 255, 0.5)')))
        fig.add_trace(go.Scatter(x=x, y=p50Lower_p, mode='lines',
                                 name='Percentiles (perturbation)',
                                 line=dict(dash='dot', color='rgba(0, 0, 255, 0.5)')))
        fig.add_trace(go.Scatter(x=x, y=p50Upper_p, mode='lines',
                                 name='Percentiles (perturbation)',
                                 line=dict(dash='dot', color='rgba(0, 0, 255, 0.5)')))
        fig.add_trace(go.Scatter(x=x, y=p95Upper_p, mode='lines',
                                 name='Percentiles (perturbation)',
                                 line=dict(dash='dot', color='rgba(0, 0, 255, 0.5)')))
    else:
        # Plot 3 samples in lightgrey
        np.random.seed(0)
        samples = np.random.multivariate_normal(mean, data['cov-temporal'], size=3)
        for i, s in enumerate(samples):
            if i == 0:
                fig.add_trace(go.Scatter(x=x, y=s, line=dict(color='grey'), name="Sample"))
            else:
                fig.add_trace(go.Scatter(x=x, y=s, line=dict(color='grey'), showlegend=False))

    return fig

@callback(
    Output('fig-expectation-value', 'figure'),
    Input('store-data', 'data'),
    Input('threshold-slider', 'value'),
    Input('fig-expectation-value', 'clickData'),
    Input('switch-distance-noise', 'on'),
    Input('switch-global-correlations', 'on'),
    Input('dropdown-selection', 'value'),
    Input('dropdown-view', 'value')
)
def update_fig_expectation(data, threshold, clickData, distanceToNoise, globalCorrelations, selection, viewSelection):
    if not data or not 'wavelet-expectation' in data:
        raise PreventUpdate
    fig = go.Figure()
    N = len(data['wavelet-distance'][0])
    if data['showDate']:
        timescale = pd.date_range(start=data['startDate'], periods=N, freq=data['freq'])
    else:
        timescale = np.arange(N)*data['dt']
    title = ''
    if distanceToNoise:
        title = 'Percentage Surpassing the 95th Noise Percentile'
        if clickData and threshold:
            fig.add_trace(go.Heatmap(z=data['wavelet-distance'], x=timescale, colorscale='Greys', showscale=False))
        else:
            fig.add_trace(go.Heatmap(z=data['wavelet-distance'], x=timescale, colorscale='Greys',
                                         colorbar={"title": 'Percentage'}))
    # fig.add_trace(go.Heatmap(z=data['wavelet-expectation'], x=timescale, y=data['wavelet-scales'], colorscale='Greys'))
    else:
        title = viewSelection + ' of Wavelet Spectrum'
        scalarField = data['wavelet-expectation']
        colorBarTitle = viewSelection
        if viewSelection == '50% Range':
            scalarField = np.array(data['wavelet-percentiles'][3]) - np.array(data['wavelet-percentiles'][1])
        elif viewSelection == '95% Range':
            scalarField = np.array(data['wavelet-percentiles'][4]) - np.array(data['wavelet-percentiles'][0])
        elif viewSelection == '2.5th Percentile':
            scalarField = data['wavelet-percentiles'][0]
        elif viewSelection == '25th Percentile':
            scalarField = data['wavelet-percentiles'][1]
        elif viewSelection == 'Median':
            scalarField = data['wavelet-percentiles'][2]
        elif viewSelection == '75th Percentile':
            scalarField = data['wavelet-percentiles'][3]
        elif viewSelection == '97.5th Percentile':
            scalarField = data['wavelet-percentiles'][4]
        zmin = 0
        zmax = np.max(np.array(scalarField)[np.logical_not(np.array(data['coiFiltering'], dtype='bool'))])
        #print("zmax: " + str(zmax))
        if clickData and threshold and selection == "Correlation":
            fig.add_trace(go.Heatmap(z=scalarField, x=timescale, colorscale='Greys', showscale=False, zmin=zmin, zmax = zmax))
        else:
            fig.add_trace(go.Heatmap(z=scalarField, x=timescale, colorscale='Greys', zmin=zmin, zmax = zmax,
                                         colorbar={"title": {"text": colorBarTitle, "side": 'right'}}))
            fig.update_coloraxes(colorbar_title_side='right')
    if threshold and threshold > 0:
        if distanceToNoise:
            binaryData = np.array(data['wavelet-distance']) > threshold
        else:
            binaryData = np.array(data['wavelet-expectation']) > threshold
        binaryData = binaryData.astype(int)
        colorscale = [[0,'black'], [1,'black']]
        #fig.add_trace(go.Heatmap(z=binaryData, y=data['wavelet-scales']))
        fig.add_trace(go.Contour(z=binaryData,
                     # y=data['wavelet-scales'],
                      x=timescale,
                      contours_coloring='lines',
                      line_width=2,
                      colorscale=colorscale,
                      contours=dict(
                          start=0.5,
                          end = 0.5
                      ), showscale=False)
                      )
        if clickData and np.count_nonzero(binaryData) > 0 and not globalCorrelations and selection=="Correlation":
            pos = (clickData['points'][0]['x'], clickData['points'][0]['y'])
            # pos_idx = (int(pos[0]/data['dt']), np.abs(np.array(data['wavelet-scales']) - pos[1]).argmin())
            # pos_idx = (int(pos[0]/data['dt']), pos[1])
            pos_idx = (np.argwhere(timescale == pos[0])[0,0], pos[1])
            if binaryData[pos_idx[1], pos_idx[0]] > 0:
                corr = np.zeros(binaryData.shape)
                N = len(data['mean-temporal'])
                dt = data['dt']
                s_0 = 2*dt
                dj = data['dj']
                A_ns = 1/np.sqrt(N)*np.array([uafourier.create_wavelet(s_0 * math.pow(2, pos_idx[1] * dj), uafourier.wk(j, N, dt), dt).conjugate()*uafourier.inverse_fourier(pos_idx[0], j, N) for j in range(int(N/2))], dtype=complex)
                sigma2 = np.sqrt(np.mean(np.diagonal(data['cov-temporal'])))
                # First multiplication, because for selected point always the same
                gammaLeft = np.dot(A_ns, (np.array(data['fftGamma-real'])[:int(N/2), :int(N/2)] + 1j*np.array(data['fftGamma-imag'])[:int(N/2), :int(N/2)])/sigma2**2)
                CLeft = np.dot(A_ns, (np.array(data['fftC-real'])[:int(N/2), :int(N/2)] + 1j*np.array(data['fftC-imag'])[:int(N/2), :int(N/2)])/sigma2**2)
                std_ns = np.sqrt(data['wavelet-var'][pos_idx[1]][pos_idx[0]])
                waveletMuIm_s = np.array(data['wavelet-expectation-imag'])[pos_idx[1], pos_idx[0]]
                waveletMuRe_s = np.array(data['wavelet-expectation-real'])[pos_idx[1], pos_idx[0]]
                for s in range(corr.shape[0]):
                    mask = np.logical_and(binaryData[s] > 0, np.logical_not(np.array(data['coiFiltering'])[s].astype(bool)))
                    if np.any(mask):
                        A_ns_dash = 1/np.sqrt(N)*np.array([[uafourier.create_wavelet(s_0 * math.pow(2, s * dj), uafourier.wk(j, N, dt), dt).conjugate()*uafourier.inverse_fourier(i, j, N) for j in range(int(N/2))] for i in range(N)], dtype=complex)
                        std_ns_dash = np.sqrt(np.array(data['wavelet-var'])[s, mask])
                        gammaDash = np.dot(gammaLeft, A_ns_dash.conj().T)[mask]
                        cDash = np.dot(CLeft, A_ns_dash.T)[mask]
                        cov_re_re = 0.5 * np.real(gammaDash + cDash)
                        cov_im_im = 0.5 * np.real(gammaDash - cDash)
                        cov_re_im = 0.5 * np.imag(gammaDash + cDash)
                        cov_im_re = -0.5 * np.imag(gammaDash - cDash)
                        waveletMuRe_s_dash = np.array(data['wavelet-expectation-real'])[s, mask]
                        waveletMuIm_s_dash = np.array(data['wavelet-expectation-imag'])[s, mask]
                        cov_re_re_square = 2 * cov_re_re ** 2 + 4 * waveletMuRe_s_dash * waveletMuRe_s * cov_re_re
                        cov_im_im_square = 2 * cov_im_im ** 2 + 4 * waveletMuIm_s_dash * waveletMuIm_s * cov_im_im
                        cov_re_im_square = 2 * cov_re_im ** 2 + 4 * waveletMuIm_s_dash * waveletMuRe_s * cov_re_im
                        cov_im_re_square = 2 * cov_im_re ** 2 + 4 * waveletMuRe_s_dash * waveletMuIm_s * cov_im_re
                        cov = cov_re_re_square + cov_im_im_square + cov_re_im_square + cov_im_re_square
                        corr[s][mask] = cov.flatten()/(std_ns * std_ns_dash)
                corr[corr==0] = np.nan
                fig.add_trace(go.Heatmap(z=corr,
                                         #y=data['wavelet-scales'],
                                         x=timescale,
                                         colorscale="RdBu",
                                         zmin = -1,
                                         zmax = 1,
                                         colorbar={"title": 'Correlation'})
                              )
                fig.add_trace(go.Scatter(
                    x=[pos[0]],
                    y=[pos_idx[1]],
                    mode='markers',
                    marker=dict(symbol='x', size=10, color='red'),
                    hoverinfo='skip',
                    showlegend=False
                ))
        elif globalCorrelations and np.count_nonzero(binaryData) > 0:
            # Compute global correlation
            mask = np.logical_and(binaryData > 0, np.logical_not(np.array(data['coiFiltering']).astype(bool)))
            numberPoints = np.count_nonzero(mask)
            corrMatrix = np.zeros((numberPoints, numberPoints))
            sCounter = 0
            dj = data['dj']
            dt = data['dt']
            s_0 = 2*dt
            for s in range(binaryData.shape[0]):
                maskS = mask[s]
                numEntriesS = np.count_nonzero(maskS)
                if np.count_nonzero(numEntriesS) > 0:
                    # Create Ans and first multiplication
                    A_s = 1 / np.sqrt(N) * np.array([[uafourier.create_wavelet(s_0 * math.pow(2, s * dj), uafourier.wk(j, N, dt), dt).conjugate() * uafourier.inverse_fourier(i, j,
                                                                                                                      N) for
                                                    j in range(int(N / 2))] for i in np.argwhere(maskS>0).flatten()], dtype=complex)
                    sigma2 = np.sqrt(np.mean(np.diagonal(data['cov-temporal'])))
                    gammaLeft = np.dot(A_s, (np.array(data['fftGamma-real'])[:int(N/2), :int(N/2)] + 1j*np.array(data['fftGamma-imag'])[:int(N/2), :int(N/2)])/sigma2**2)
                    CLeft = np.dot(A_s, (np.array(data['fftC-real'])[:int(N/2), :int(N/2)] + 1j*np.array(data['fftC-imag'])[:int(N/2), :int(N/2)])/sigma2**2)
                    sDashCounter = 0
                    waveletMuIm_s = np.array(data['wavelet-expectation-imag'])[s, maskS]
                    waveletMuRe_s = np.array(data['wavelet-expectation-real'])[s, maskS]
                    std_s = np.sqrt(np.array(data['wavelet-var'])[s, maskS])
                    for sDash in range(binaryData.shape[0]):
                        # Create AnDashSDash and first multiplication
                        maskSDash = mask[sDash]
                        numEntriesSDash = np.count_nonzero(maskSDash)
                        if np.count_nonzero(maskSDash) > 0:
                            A_sDash = 1 / np.sqrt(N) * np.array([[uafourier.create_wavelet(s_0 * math.pow(2, sDash * dj), uafourier.wk(j, N, dt), dt).conjugate() * uafourier.inverse_fourier(i, j,
                                                                                                                              N) for
                                                            j in range(int(N / 2))] for i in np.argwhere(maskSDash>0).flatten()], dtype=complex)
                            gammaDash = np.dot(gammaLeft, A_sDash.conj().T)
                            cDash = np.dot(CLeft, A_sDash.T)
                            cov_re_re = 0.5 * np.real(gammaDash + cDash)
                            cov_im_im = 0.5 * np.real(gammaDash - cDash)
                            cov_re_im = 0.5 * np.imag(gammaDash + cDash)
                            cov_im_re = -0.5 * np.imag(gammaDash - cDash)
                            waveletMuRe_s_dash = np.array(data['wavelet-expectation-real'])[sDash, maskSDash]
                            waveletMuIm_s_dash = np.array(data['wavelet-expectation-imag'])[sDash, maskSDash]
                            cov_re_re_square = 2 * cov_re_re ** 2 + 4 * np.outer(waveletMuRe_s, waveletMuRe_s_dash) * cov_re_re
                            cov_im_im_square = 2 * cov_im_im ** 2 + 4 * np.outer(waveletMuIm_s, waveletMuIm_s_dash) * cov_im_im
                            cov_re_im_square = 2 * cov_re_im ** 2 + 4 * np.outer(waveletMuRe_s, waveletMuIm_s_dash) * cov_re_im
                            cov_im_re_square = 2 * cov_im_re ** 2 + 4 * np.outer(waveletMuIm_s, waveletMuRe_s_dash) * cov_im_re
                            cov = cov_re_re_square + cov_im_im_square + cov_re_im_square + cov_im_re_square
                            std_s_dash = np.sqrt(np.array(data['wavelet-var'])[sDash, maskSDash])
                            subCorrMatrix = cov/np.outer(std_s, std_s_dash)
                            corrMatrix[sCounter:sCounter+numEntriesS, sDashCounter:sDashCounter+numEntriesSDash] = subCorrMatrix
                            sDashCounter += numEntriesSDash
                sCounter += numEntriesS
            # Embed/compute colors
            positions, eigenvalues = uafourier.mds(corrMatrix, dimensions=2)
            np.save("positions.npy", positions)
            plt.scatter(positions[0], positions[1])
            plt.show()
            colors = uafourier.getColors(positions)
            globalCorr = np.full((binaryData.shape[0], binaryData.shape[1], 4), 0.0)
            indices = np.argwhere(mask)
            globalCorr[indices[:,0], indices[:,1],3] = 1.0
            globalCorr[indices[:,0], indices[:,1],:3] = colors
            # Show colors
            pil_image = Image.fromarray((globalCorr[::-1] * 255).astype(np.uint8))
            width = timescale[-1]-timescale[0]#np.max(timescale)
            if data['showDate']:
                width=(timescale[-1]-timescale[0])
                width = width.total_seconds()*1000
            fig.add_layout_image(
                dict(
                    source=pil_image,
                    xref="x",
                    yref="y",
                    x=timescale[0],
                    y=-0.5,
                    sizex=width,
                    sizey=pil_image.height,
                    sizing="stretch",
                    # opacity=0.5,
                    yanchor='bottom',
                    xanchor='left',
                    layer="above")
            )
    s_values = data['sValues']
    s_max = 1/data['dj']*np.log2(np.max(data['wavelet-scales'])/(2*data['dt']))
    fig.add_trace(go.Scatter(x=timescale, y=s_values, mode='lines', name='COI', showlegend=False,
                             line=dict(color='lightgrey'), hoverinfo='skip'))
    # Gray out above COI
    fig.add_trace(go.Scatter(x=np.concatenate([timescale, timescale[::-1]]), y=np.concatenate([np.ones(len(s_values))*s_max, s_values[::-1]]),
                             fill='toself', fillcolor='lightgrey', mode='none', opacity=0.5, showlegend=False, hoverinfo='skip'))
    if clickData:
        pos = (clickData['points'][0]['x'], clickData['points'][0]['y'])
        if selection == 'Slice Time':
            fig.add_vline(x=pos[0], line=dict(color="red", width=2))
        if selection == 'Slice Scale':
            fig.add_hline(y=pos[1], line=dict(color="red", width=2))

    formatted_numbers = ['{:.3f}'.format(num) for num in data['wavelet-scales']]
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Scale',
        margin=dict(l=40, r=0, t=25, b=0),
        title=title,
        template=TEMPLATE,
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(data['wavelet-scales']))[::4],
            ticktext=formatted_numbers[::4],
            range=[0, np.max(s_values)+1]
        )
    )
    if data['showDate']:
        fig.update_layout(
            xaxis=dict(
                type='date'
            )
        )
    return fig

@callback(
    Output('store-perturbation', 'data'),
    Input('tabs', 'value'),
    Input('dropdown-distribution-selection', 'value'),
    Input('slider1', 'value'),
    Input('sliderVal', 'value'),
    Input('slider2', 'value'),
    Input('slider3', 'value'),
    State('store-data', 'data')
)
def update_perturbation(tab, dropdown, slider1, sliderVal, slider2, slider3, data):
    if tab == 'percentiles' or not 'mean-temporal' in data:
        raise PreventUpdate
    dataPert = {}
    theta = None
    if dropdown == 'uniform':
        theta = np.zeros((len(data['mean-temporal']),len(data['mean-temporal'])))
        np.fill_diagonal(theta, sliderVal)
    elif dropdown == 'normal':
        points = np.linspace(0, len(data['mean-temporal'])*data['dt'], len(data['mean-temporal']))
        xx, yy = np.meshgrid(points,points)
        # Compute pairwise differences between points
        pairwise_diff = (xx - slider1)**2+ (yy - slider1) ** 2
        # Compute covariance matrix using the Gaussian kernel
        theta = slider3*np.exp(-0.5 * ((pairwise_diff)) / slider2)/(2*np.pi*slider2)
    else:
        print(dropdown)
    dataPert['theta'] = theta
    _, gamma, c = uafourier.ua_fourier(np.zeros(len(theta[0])), theta)
    dataPert['fft_gamma_off-real'] = np.real(gamma)
    dataPert['fft_gamma_off-im'] = np.imag(gamma)
    dataPert['fft_c_off-real'] = np.real(c)
    dataPert['fft_c_off-im'] = np.imag(c)
    dt = data['dt']
    dj = data['dj']
    s_0 = 2 * dt
    J = int(math.log2(len(theta[0]) * dt / s_0) / dj)
    waveletExpectationDiff = np.empty((J, len(theta[0])))
    waveletVarDiff = np.empty((J, len(theta[0])))
    mean_real = np.array(data['wavelet-expectation-real'])
    mean_imag = np.array(data['wavelet-expectation-imag'])
    cov_re_re = np.array(data['wavelet-var-re-re'])
    cov_re_im = np.array(data['wavelet-var-re-im'])
    cov_im_im = np.array(data['wavelet-var-im-im'])
    for i in range(J):
        print(i)
        _, waveletGamma, waveletC = uafourier.wavelet(np.zeros(len(theta[0])), gamma, c, s_0 * math.pow(2, i * dj), dt)
        var_re_re = np.real(np.diagonal(waveletGamma+waveletC))
        var_im_im = np.real(np.diagonal(waveletGamma-waveletC))
        var_re_im = np.imag(np.diagonal(waveletGamma+waveletC))
        waveletExpectationDiff[i] = var_re_re + var_im_im
        waveletVarInter = 2 * (cov_re_re[i] + mean_real[i] ** 2 * var_re_re + 2 * cov_re_im[i] + 2 * mean_real[i] * mean_imag[i] * var_re_im + cov_im_im[i] + mean_imag[i] ** 2 + var_im_im)
        waveletVarDiff[i] = 2*(var_re_re+2*var_re_im + var_im_im) + waveletVarInter
    dataPert['wavelet-expectation-diff'] = waveletExpectationDiff
    dataPert['wavelet-var-diff'] = waveletVarDiff
    return dataPert

@callback(
    Output('fig-expectation-value-sensitivity', 'figure'),
    Input('store-perturbation', 'data'),
    Input('store-data', 'data'),
    Input('switch-relative-difference', 'on')
)
def update_wavelet_expectation_sensitivity(dataPert, data, relative):
    if not data or not dataPert:
        raise PreventUpdate
    fig = go.Figure()
    waveletExpDiff = np.array(dataPert['wavelet-expectation-diff'])
    if relative:
        waveletExpDiff /= np.array(data['wavelet-expectation'])
    # fig.add_trace(go.Heatmap(z=waveletExpDiff, y=data['wavelet-scales'], colorscale='Reds'))
    N = len(waveletExpDiff[0])
    if data['showDate']:
        timescale = pd.date_range(start=data['startDate'], periods=N, freq=data['freq'])
    else:
        timescale = np.arange(N)*data['dt']
    fig.add_trace(go.Heatmap(x=timescale, z=waveletExpDiff, colorscale='Reds'))
    s_values = data['sValues']
    s_max = 1 / data['dj'] * np.log2(np.max(data['wavelet-scales']) / (2 * data['dt']))
    fig.add_trace(go.Scatter(x=timescale, y=s_values, mode='lines', name='COI', showlegend=False,
                             line=dict(color='lightgrey'), hoverinfo='skip'))
    # Gray out above COI
    fig.add_trace(go.Scatter(x=np.concatenate([timescale, timescale[::-1]]),
                             y=np.concatenate([np.ones(len(s_values)) * s_max, s_values[::-1]]),
                             fill='toself', fillcolor='lightgrey', mode='none', opacity=0.5, showlegend=False,
                             hoverinfo='skip'))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Scale',
        margin=dict(l=40, r=0, t=25, b=0),
        title='Change in Expected Value of Wavelet Spectrum',
        template=TEMPLATE
    )
    return fig

@callback(
    Output('fig-wavelet-variance-sensitivity', 'figure'),
    Input('store-perturbation', 'data'),
    Input('store-data', 'data'),
    Input('switch-relative-difference', 'on')
)
def update_wavelet_var_sensitivity(dataPert, data, relative):
    if not data or not dataPert:
        raise PreventUpdate
    fig = go.Figure()
    waveletVarDiff = np.array(dataPert['wavelet-var-diff'])
    if relative:
        waveletVarDiff /= np.array(data['wavelet-var'])
    # fig.add_trace(go.Heatmap(z=waveletVarDiff, y=data['wavelet-scales'], colorscale='Reds'))
    N = len(waveletVarDiff[0])
    if data['showDate']:
        timescale = pd.date_range(start=data['startDate'], periods=N, freq=data['freq'])
    else:
        timescale = np.arange(N)*data['dt']
    fig.add_trace(go.Heatmap(x=timescale, z=waveletVarDiff, colorscale='Reds'))
    s_values = data['sValues']
    s_max = 1 / data['dj'] * np.log2(np.max(data['wavelet-scales']) / (2 * data['dt']))
    fig.add_trace(go.Scatter(x=timescale, y=s_values, mode='lines', name='COI', showlegend=False,
                             line=dict(color='lightgrey'), hoverinfo='skip'))
    # Gray out above COI
    fig.add_trace(go.Scatter(x=np.concatenate([timescale, timescale[::-1]]),
                             y=np.concatenate([np.ones(len(s_values)) * s_max, s_values[::-1]]),
                             fill='toself', fillcolor='lightgrey', mode='none', opacity=0.5, showlegend=False,
                             hoverinfo='skip'))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Scale',
        margin=dict(l=40, r=0, t=25, b=0),
        title='Change in Variance of Wavelet Spectrum',
        template=TEMPLATE
    )
    return fig

@callback(
    Output('fig-fourier-transform', 'figure'),
    Input('store-data', 'data'),
    Input('switch-distance-noise', 'on')
)
def update_fig_fourier(data, noise):
    # Show median, 50% quantile and 95% quantile
    if not data or not 'percentiles-fourier' in data:
        raise PreventUpdate

    percentiles = data['percentiles-fourier']

    x = (np.arange(len(percentiles[0])) + 2) / len(percentiles[0]) / (2*data
    ['dt'])

    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=percentiles[2], mode='lines', name='Median'))

    # Plotting the 50% percentile lines (mean ± 0.6745*standard deviation)
    percentile50lower = percentiles[1]
    percentile50upper = percentiles[3]
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([percentile50lower, percentile50upper[::-1]]),
                             fill='toself',
                             fillcolor='rgba(0, 0, 255, 0.3)',  # Blue with low opacity
                             line=dict(color='rgba(255, 255, 255, 0)'), name='50%'))

    # Plotting the 95% percentile lines (mean ± 1.96*standard deviation)
    percentile95lower = percentiles[0]
    percentile95upper = percentiles[4]
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([percentile95lower, percentile95upper[::-1]]),
                             fill='toself',
                             fillcolor='rgba(0, 0, 255, 0.1)',  # Blue with lower opacity
                             line=dict(color='rgba(255, 255, 255, 0)'), name='95%'))

    if noise:
        fig.add_trace(go.Scatter(x=x, y=np.array(data['noise-spectrum'])*5.99/2, mode='lines', name='Noise spectrum', line=dict(color='black', dash='dash')))

    fig.update_layout(
        xaxis_title='Frequency',
        yaxis_title='Energy Density',
        margin=dict(l=40, r=0, t=25, b=0),
        title='Fourier Transformation',
        template=TEMPLATE
    )
    return fig

@callback(
    Output('fig-distribution-vis', 'figure'),
    Input('store-data', 'data'),
    Input('fig-expectation-value', 'clickData'),
    Input('dropdown-selection', 'value'),
    Input('switch-distance-noise', 'on')
)
def update_distribution_vis(data, clickData, selection, noise):
    if not 'wavelet-percentiles' in data or not 'coiFiltering' in data or not clickData or selection=='Correlation':
        raise PreventUpdate
    fig = go.Figure()
    percentiles = np.array(data['wavelet-percentiles'])
    if len(percentiles) != 5:
        raise PreventUpdate
    y = np.arange(len(data['wavelet-scales']))
    formatted_numbers = ['{:.3f}'.format(num) for num in data['wavelet-scales']]
    pos = (clickData['points'][0]['x'], clickData['points'][0]['y'])
    if data['showDate']:
        x = pd.date_range(start=data['startDate'], periods=len(percentiles[0][0]), freq=data['freq'])
    else:
        x = np.arange(len(percentiles[0][0]))*data['dt']
    if selection == 'Slice Time':
        time_idx = np.argwhere(pos[0] == x)[0,0]
        percentiles = percentiles[:,:,time_idx]
        fig.add_trace(go.Scatter(y=y, x=percentiles[2], mode='lines', name='Median'))

        # Plotting the 50% percentile lines (mean ± 0.6745*standard deviation)
        percentile50lower = percentiles[1]
        percentile50upper = percentiles[3]
        fig.add_trace(
            go.Scatter(y=np.concatenate([y, y[::-1]]), x=np.concatenate([percentile50lower, percentile50upper[::-1]]),
                       fill='toself',
                       fillcolor='rgba(0, 0, 255, 0.3)',  # Blue with low opacity
                       line=dict(color='rgba(255, 255, 255, 0)'), name='50%'))

        # Plotting the 95% percentile lines (mean ± 1.96*standard deviation)
        percentile95lower = percentiles[0]
        percentile95upper = percentiles[4]
        fig.add_trace(
            go.Scatter(y=np.concatenate([y, y[::-1]]), x=np.concatenate([percentile95lower, percentile95upper[::-1]]),
                       fill='toself',
                       fillcolor='rgba(0, 0, 255, 0.1)',  # Blue with lower opacity
                       line=dict(color='rgba(255, 255, 255, 0)'), name='95%'))
        # Add gray overlay for COI
        s = data['sValues'][time_idx]
        fig.add_shape(type="rect",
                      y0=s, x0=0, y1=max(y), x1=np.max(percentiles),
                      line=dict(color="gray", width=0), fillcolor="lightgray", opacity=0.5
                      )
        if noise:
            s_0 = 2*data['dt']
            scale = s_0 * np.power(2, np.arange(len(y)) * data['dj'])
            w_0 =6
            beta = 4*np.pi/(w_0+np.sqrt(2+w_0**2))
            k = np.round(1 / (beta * scale) * len(x) * data['dt'])
            k = k.astype(int)
            p = np.array(data['noise-spectrum'])[k]*5.99/2
            fig.add_trace(go.Scatter(x=p, y=y, mode='lines', name='Noise spectrum', line=dict(color='black', dash='dash')))
        fig.update_layout(
            yaxis = dict(
                tickmode='array',
                tickvals=np.arange(len(data['wavelet-scales']))[::4],
                ticktext=formatted_numbers[::4],
                range=[0, np.max(data['sValues']) + 1]
            ),
            xaxis_title='Wavelet Energy',
            yaxis_title='Scale'
        )
    elif selection == 'Slice Scale':
        percentiles = percentiles[:,pos[1]]
        fig.add_trace(go.Scatter(x=x, y=percentiles[2], mode='lines', name='Median'))

        # Plotting the 50% percentile lines (mean ± 0.6745*standard deviation)
        percentile50lower = percentiles[1]
        percentile50upper = percentiles[3]
        fig.add_trace(
            go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([percentile50lower, percentile50upper[::-1]]),
                       fill='toself',
                       fillcolor='rgba(0, 0, 255, 0.3)',  # Blue with low opacity
                       line=dict(color='rgba(255, 255, 255, 0)'), name='50%'))

        # Plotting the 95% percentile lines (mean ± 1.96*standard deviation)
        percentile95lower = percentiles[0]
        percentile95upper = percentiles[4]
        fig.add_trace(
            go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([percentile95lower, percentile95upper[::-1]]),
                       fill='toself',
                       fillcolor='rgba(0, 0, 255, 0.1)',  # Blue with lower opacity
                       line=dict(color='rgba(255, 255, 255, 0)'), name='95%'))
        if data['showDate']:
            timespan = relativedelta(days = int(np.sqrt(2)*data['sValues'][pos[1]]*30))
            if data['freq'] == 'D':
                timespan = relativedelta(days=int(np.sqrt(2) * data['sValues'][pos[1]]))
            fig.add_shape(type="rect",
                          y0=0, x0=x[0], y1=np.max(percentiles), x1=x[0] + timespan,
                          line=dict(color="gray", width=0), fillcolor="lightgray", opacity=0.5
                          )
            fig.add_shape(type="rect",
                          y0=0, x0=max(x), y1=np.max(percentiles), x1=max(x)-timespan,
                          line=dict(color="gray", width=0), fillcolor="lightgray", opacity=0.5
                          )
        else:
            timespan = np.sqrt(2) * data['sValues'][pos[1]]*data['dt']
            fig.add_shape(type="rect",
                          y0=x[0], x0=0, y1=np.max(percentiles), x1=x[0] + timespan,
                          line=dict(color="gray", width=0), fillcolor="lightgray", opacity=0.5
                          )
            fig.add_shape(type="rect",
                          y0=0, x0=max(x), y1=np.max(percentiles), x1=max(x) - timespan,
                          line=dict(color="gray", width=0), fillcolor="lightgray", opacity=0.5
                          )

        if noise:
            s_0 = 2*data['dt']
            scale = s_0 * math.pow(2, pos[1] * data['dj'])
            w_0 =6
            beta = 4*np.pi/(w_0+np.sqrt(2+w_0**2))
            k = int(round(1 / (beta * scale) * len(x) * data['dt']))
            fig.add_trace(go.Scatter(x=x, y=np.ones(len(x))*np.array(data['noise-spectrum'])[k]*5.99/2, mode='lines', name='Noise spectrum', line=dict(color='black', dash='dash')))
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Wavelet Energy'
        )


    fig.update_layout(
        margin=dict(l=40, r=0, t=25, b=0),
        template=TEMPLATE
    )
    return fig

@callback(
    Output('fig-fourier-transform-sensitivity', 'figure'),
    Input('store-perturbation', 'data'),
    Input('store-data', 'data')
)
def update_fig_fourier_sensitivity(dataPert, data):
    if not dataPert:
        raise PreventUpdate
    fig = go.Figure()
    mean = np.array(data['fftMean-real']) + 1j*np.array(data['fftMean-im'])
    gamma = np.array(data['fftGamma-real']) + 1j*np.array(data['fftGamma-imag'])
    c = np.array(data['fftC-real']) + 1j*np.array(data['fftC-imag'])
    gammaP = np.array(dataPert['fft_gamma_off-real']) + 1j*np.array(dataPert['fft_gamma_off-im'])
    cP = np.array(dataPert['fft_c_off-real']) + 1j*np.array(dataPert['fft_c_off-im'])
    meanFFT, covFFT = uafourier.energy_spectral_density(mean, gamma, c)
    meanFFT_p, covFFT_p = uafourier.energy_spectral_density(np.zeros(len(mean)), gammaP, cP)
    x = (np.arange(len(meanFFT))) / len(meanFFT) / (2 * data['dt'])
    fig.add_trace(go.Scatter(x=x, y=meanFFT, mode='lines', name='Mean', line=dict(color='rgba(255, 0, 0, 255)')))
    fig.add_trace(go.Scatter(x=x, y=meanFFT+meanFFT_p, mode='lines', name='Mean with perturbation', line=dict(dash='dot', color='rgba(255, 0, 0, 255)')))
    # Variance
    var = np.diagonal(covFFT)
    var_p = np.diagonal(covFFT_p)
    cov_re_re = 0.5*np.real(np.diagonal(gamma + c))
    cov_im_im = 0.5*np.real(np.diagonal(gamma - c))
    cov_re_im = 0.5*np.imag(np.diagonal(gamma+c))
    cov_re_re_p = 0.5*np.real(np.diagonal(gammaP + cP))
    cov_im_im_p = 0.5*np.real(np.diagonal(gammaP - cP))
    cov_re_im_p = 0.5*np.imag(np.diagonal(gammaP+cP))
    mean_real = np.real(mean)
    mean_imag = np.imag(mean)
    var_inter = 2*(cov_re_re+mean_real**2*cov_re_re_p + 2*cov_re_im + 2*mean_real*mean_imag*cov_re_im_p + cov_im_im + mean_imag**2+cov_im_im_p)[:len(meanFFT)]
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([meanFFT - np.sqrt(var), (meanFFT + np.sqrt(var))[::-1]]),
                             fill='toself',
                             fillcolor='rgba(255, 0, 0, 0.3)',  # Blue with low opacity
                             line=dict(color='rgba(255, 255, 255, 0)'), name='Standard deviation'))
    fig.add_trace(go.Scatter(x=x, y=meanFFT + np.sqrt(var + var_p+var_inter), mode='lines', name='Standard deviation (perturbation)', line=dict(dash='dot', color='rgba(255, 0, 0, 128)')))
    fig.add_trace(go.Scatter(x=x, y=meanFFT - np.sqrt(var + var_p+var_inter), mode='lines', name='Standard deviation (perturbation)', line=dict(dash='dot', color='rgba(255, 0, 0, 128)')))
    fig.update_layout(
        xaxis_title='Frequency',
        yaxis_title='Energy Density',
        margin=dict(l=40, r=0, t=25, b=0),
        title='Fourier Transformation',
        template=TEMPLATE
    )
    return fig

@callback(
    Output('dropdown-view', 'value'),
    Input("keyboard", "n_keydowns"),
    State("keyboard", "keydown"),
    State('dropdown-view', 'value')
)
def update_view_dropdown(_, event, currSelection):
    if not event:
        raise PreventUpdate
    change = 1
    if event['key'] == 'a':
        change = -1
    curr_idx = np.argwhere(np.array(options) == currSelection)[0,0]
    if curr_idx > 0 or change > 0:
        curr_idx += change
        curr_idx = curr_idx%len(options)
    else:
        curr_idx = len(options)-1
    return options[curr_idx]