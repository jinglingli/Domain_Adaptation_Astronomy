'''
Non-periodic feature set for light curves as defined in Richards et al. (2011)

Written by Matthew J. Graham, July 2011
Version: 0.13 - add LS
         0.14 - server error handling, file input/output consistent with networked
         0.15 - add GPU LS, SC
'''
from ls import fasper, getSignificance
#import matplotlib.pyplot as plt
import numpy as np
from qso_fit import qso_fit
from scipy.optimize import leastsq
import scipy.stats as stats
from StringIO import StringIO
import sys
from urllib import urlencode


def _flux_percentile_ratio(data, p1, p2, p3, p4):
  ssap = stats.scoreatpercentile                   
  flux = 10.**(-0.4 * data[1])
  ratio = (ssap(flux, p2) - ssap(flux, p1)) / (ssap(flux, p4) - ssap(flux, p3))
  return ratio


def _time_sort(data):
  data = data.transpose()
  data = data[data[:,0].argsort(),].transpose()
  return data


def amplitude(data):
  '''
  Half the difference between the maximum and minimum magnitudes
  '''
  amplitude = 0.5 * (np.max(data[1]) - np.min(data[1]))
  return amplitude


def beyond1std(data):
  '''
  Percentage of points beyond one st. dev. from the weighted mean
  '''
  if len(data) > 2:
    wmean = np.average(data[1], weights = 1./data[2]) 
  else:
    wmean = np.average(data[1])
  sd = np.std(data[1])
  beyonddist = np.where(abs(data[1] - wmean) > sd, 1, 0).sum() / float(len(data[1]))
  return beyonddist


def flux_percentile_ratio_mid20(data):
  '''
  Ratio of flux percentiles (60th - 40th) over (95th - 5th)
  '''
  ratio = _flux_percentile_ratio(data, 40, 60, 5, 95)
  return ratio


def flux_percentile_ratio_mid35(data):
  '''
  Ratio of flux percentiles (67.5th - 32.5th) over (95th - 5th)
  '''
  ratio = _flux_percentile_ratio(data, 32.5, 67.5, 5, 95)
  return ratio


def flux_percentile_ratio_mid50(data):
  '''
  Ratio of flux percentiles (75th - 25th) over (95th - 5th)
  '''
  ratio = _flux_percentile_ratio(data, 25, 75, 5, 95)
  return ratio


def flux_percentile_ratio_mid65(data):
  '''
  Ratio of flux percentiles (17.5th - 82.5th) over (95th - 5th)
  '''
  ratio = _flux_percentile_ratio(data, 17.5, 82.5, 5, 95)
  return ratio


def flux_percentile_ratio_mid80(data):
  '''
  Ratio of flux percentiles (10th - 90th) over (95th - 5th)
  '''
  ratio = _flux_percentile_ratio(data, 10, 90, 5, 95)
  return ratio


def linear_trend(data):
  '''
  Slope of a linear fit to the light-curve magnitudes
  '''
  n = len(data[0])
  m, b = np.polyfit(data[0], data[1], 1)
  return m


def max_slope(data):
  '''
  Maximum absolute flux slope between two consecutive observations
  '''
  data = _time_sort(data)
  max_slope = 0
  for i in range(len(data[0]) - 1):
    if data[0][i + 1] != data[0][i]:
      max_slope = max(max_slope, abs((data[1][i + 1] - data[1][i]) / (data[0][i + 1] - data[0][i])))
  return max_slope
                     

def median_absolute_deviation(data):
  ''' 
  Median discrepancy of the fluxes from the median flux
  '''
  medmag = np.median(data[1])
  mad = np.median(abs(data[1] - medmag))
  return mad


def median_buffer_range_percentage(data):
  '''
  Percentage of fluxes within 10% of the amplitude from the median
  '''
  medmag = np.median(data[1])
  mbrp = np.where( abs(data[1] - medmag) <= 0.1 * medmag , 1, 0).sum() / float(len(data[1]))
  return mbrp


def pair_slope_trend(data):
  '''
  Percentage of last 30 pairs of consecutive flux measurements that have positive slope
  '''
  data = _time_sort(data)
  limit = len(data[1]) > 30 and 30 or len(data[1])
  pst = 0
  if limit > 1:
    pos = 0;
    neg = 0;
    for i in range(-limit, -1):
      if data[0][i + 1] != data[0][i]:
        diff = (data[1][i + 1] - data[1][i]) / (data[0][i + 1] - data[0][i])
        if diff > 0: pos += 1
        elif diff < 0: neg += 1
    pst = (pos - neg) / float(limit - 1)
  return pst
    

def percent_amplitude(data):
  '''
  Largest percentage difference between either the max or min flux and the median
  '''
  flux = 10.**(-0.4 * data[1])
  medflux = np.median(flux)
  pa = np.max(abs(np.max(flux) - medflux), abs(np.min(flux) - medflux)) / medflux
  return pa


def percent_difference_flux_percentile(data):
  '''
  Ratio of flux percentile (95th - 5th) over the median flux
  '''
  flux = 10.**(-0.4 * data[1])
  medflux = np.median(flux)
  ssap = stats.scoreatpercentile                   
  pdfp = (ssap(flux, 95) - ssap(flux, 5)) / medflux
  return pdfp


def qso(data):
  '''
  Quasar (non-)variability metric in Butler & Bloom (2011)
  '''
  out_dict = qso_fit(data[0], data[1], data[2], filter='u')
#  return (out_dict['chi2_qso/nu'], out_dict['chi2_qso/nu_NULL'])
#  return (out_dict['chi2_qso/nu'], out_dict['chi2_qso/nu_NULL'], out_dict['signif_qso'], out_dict['signif_vary'], out_dict['class'])
  result = [out_dict['chi2_qso/nu'], out_dict['chi2_qso/nu_NULL'], out_dict['signif_qso'], out_dict['signif_vary']]
  return result


def skew(data):
  '''
  Skew of the magnitudes
  '''
  skew = stats.skew(data[1])
  return skew


def small_kurtosis(data):
  '''
  Kurtosis of the magnitudes, reliable down to a small number of epochs
  using http://www.xycoon.com/peakedness_small_sample_test_1.htm
  '''
  kurtosis = 0
  if len(data[0]) > 4:
    meanmag = np.mean(data[1])
    n = len(data[1])
    s = np.sqrt(np.power(data[1] - meanmag, 2).sum() / (n - 1))
    kurtosis = (n * (n + 1) *  np.power((data[1] - meanmag) / s, 4).sum() / (( n - 1) * (n - 2) * (n - 3))) - ((3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3)))
  return kurtosis 


def std(data): 
  '''
  Standard deviation of magnitudes
  '''
  std = np.std(data[1])
  return std


def stetson_j(data):
  '''
  Welch-Stetson variability index J (Stetson 1996) with weighting scheme from (Zhang et al. 2003)
  taking successive pairs in time-order
  '''
  data = _time_sort(data)
  n = len(data[1])
  stetson_j = 0
  if n > 1:
    if len(data) > 2: 
      delta = np.sqrt(float(n) / (n - 1)) * (data[1] - np.mean(data[1])) / data[2] 
    else:
      delta = np.sqrt(float(n) / (n - 1)) * (data[1] - np.mean(data[1])) 
    sum = 0.
    w_sum = 0.
    dt = 0.
    for i in range(n - 1):
      dt += (data[0][i + 1] - data[0][i])
    dt = dt / float(n - 1)
    for i in range(n - 1):
      wk = np.exp(-(data[0][i + 1] - data[0][i]) / dt)
      pk = delta[i] * delta[i + 1]
      sum += wk * np.sign(pk) * np.sqrt(abs(pk))
      w_sum += wk
    stetson_j = sum / w_sum
  return stetson_j


def stetson_k(data):
  '''
  Welch-Stetson variability index K (Stetson 1996)
  '''
  n = len(data[1])
  stetson_k = 0
  if n > 1:
    if len(data) > 2:
      delta = np.sqrt(float(n) / (n - 1)) * (data[1] - np.mean(data[1])) / data[2] 
    else:
      delta = np.sqrt(float(n) / (n - 1)) * (data[1] - np.mean(data[1]))
    top = abs(delta).sum() / n
    bottom = np.sqrt((delta * delta).sum() / n)
    stetson_k = top / bottom
  return stetson_k


def ls(data):
  '''
  Lomb-Scargle periodogram
  '''
  period = 0
  if len(data[1]) > 1:
    (wk1, wk2, nout, jmax, prob) = fasper(data[0], data[1], 4, 100)
#  return (1./wk1[jmax], prob)
    period = 1./wk1[jmax]
  return period


def _sigma_clip(data, sigma):
  '''
  Sigma clip data
  '''
  wmean = np.average(data[1], weights = 1./data[2])
  sd = np.std(data[1])
  index = np.where(abs(data[1] - wmean) < sigma * sd)
  return data.transpose()[index].transpose()


def gpuls(data, sigma = 0, exclude = None, periods = 5, plot = False):
  '''
  GPU version of Lomb-Scargle periodogram
  '''
  lsp = Culsp()
  if sigma > 0: data = _sigma_clip(data, sigma)
  if plot:
    plt.plot(data[0], data[1], 'x')
    plt.axis([data[0].min(), data[0].max(), data[1].min() - 0.5, data[1].max() + 0.5])
    plt.show()
  rawdata = data.copy()
  data = lsp.normalize_data(_time_sort(data))
  dt = np.array([data[0][i + 1] - data[0][i] for i in range(len(data[0]) - 1)])
  dt_med = np.median(dt)
  over = 10
  hif = (data[0].max() - data[0].min()) / (dt_med * len(data[0]))
  period = lsp.culsp(data, over, hif, exclude, periods)
  if plot:
    # Phase diagram
    phase = (rawdata[0] - rawdata[0].min()) / period[0][0]
    phase -= np.floor(phase)
    plt.plot(np.concatenate((phase, phase + 1)), np.concatenate((rawdata[1], rawdata[1])), 'x')
    plt.axis([0, 2, rawdata[1].max() + 0.5, rawdata[1].min() - 0.5])
    plt.show()
  return period


def aov(data):
  '''
  Periods according to Schwarzenberg-Czerny (1989)
  '''
  data = _time_sort(data)
  period = av.aov(data.T)
  return period


def gpugls(data, sigma = 0, exclude = None, periods = 5, plot = False):
  '''
  GPU version of generalized Lomb-Scargle periodogram
  '''
  lsp = Cuglsp()
  if sigma > 0: data = _sigma_clip(data, sigma)
  if plot:
    plt.plot(data[0], data[1], 'x')
    plt.axis([data[0].min(), data[0].max(), data[1].min() - 0.5, data[1].max() + 0.5])
    plt.show()
  rawdata = data.copy()
#  data = lsp.normalize_data(_time_sort(data))
  data = _time_sort(data)
  dt = np.array([data[0][i + 1] - data[0][i] for i in range(len(data[0]) - 1)])
  dt_med = np.median(dt)
  over = 10
  hif = (data[0].max() - data[0].min()) / (dt_med * len(data[0]))
  period = lsp.cuglsp(data, over, hif, exclude, periods)
  if plot:
    # Phase diagram
    phase = (rawdata[0] - rawdata[0].min()) / period[0][0]
    phase -= np.floor(phase)
    plt.plot(np.concatenate((phase, phase + 1)), np.concatenate((rawdata[1], rawdata[1])), 'x')
    plt.axis([0, 2, rawdata[1].max() + 0.5, rawdata[1].min() - 0.5])
    plt.show()
  return period


def pdm(data):
  '''
  Periods according to phase dispersion minimization (1978)
  '''
  data = _time_sort(data)
  period = pd.pdm2(data.T)
  return period


def fastchi(data):
  '''
  Periods according to fast chi-squared (Palmer 2009)
  '''
  data = _time_sort(data)
  period = fcs.runchi(data.T)
  return period


def sc(data, bindensity = 10.):
  '''
  Self-correlation
  '''
  data = _time_sort(data)
  raw = [(data[0][j] - data[0][i], data[1][j] - data[1][i]) for i in range(len(data[0])) for j in range(len(data[0])) if j > i]
  raw = np.array(raw).transpose()
  dt = bindensity * raw[0].max() / len(raw[0])
  raw[0] = np.floor(raw[0] / dt)
  dtdm = [(i, raw[1][np.where(raw[0] == i)].mean()) for i in range(int(raw[0].max()) + 1)]
  dtdm = np.array([x for x in dtdm if not np.isnan(x[1])]).transpose()
  dtdm[0] = (dtdm[0] + 0.5) * dt
  return dtdm.transpose()


def sf(data, dt = 0.1, logt = True):
  '''
  First-order structure function using squared diffs.
  '''
  data = _time_sort(data)
  raw = [(data[0][j] - data[0][i], data[1][j] - data[1][i]) for i in range(len(data[0])) for j in range(len(data[0])) if j > i]
  raw = np.array(raw).transpose()
  if logt:
    raw = raw[:,~(raw[0] == 0)]
    raw[0] = np.floor(np.log10(raw[0]) / dt)
  else:
    raw[0] = np.floor(raw[0] / dt)
  raw[1] *= raw[1] 
  dtdm = [(i, raw[1][np.where(raw[0] == i)].mean()) for i in range(int(raw[0].min()), int(raw[0].max()) + 1)]
  dtdm = np.array([x for x in dtdm if not np.isnan(x[1])]).transpose()
  dtdm[0] = (dtdm[0] + 0.5) * dt
  return dtdm.transpose()


def freqparam(data):
  '''
  First three frequencies and their first four harmonics, the offset and ratio of variances for
  first prewhitened to original spectrum
  '''
  data = _time_sort(data)
  freqs = fp.freqparam(data)
  params = [freqs[0]]
  for i in range(3):
    params.append(freqs[1][i])
    for j in range(4):
      params.append(freqs[2][i][j])
      params.append(freqs[3][i][j])
  params.append(freqs[4])
  return params


def rcorbor(data):
  '''
  Find R Cor Bor-type objects: fraction of points 1.5 mag below the median
  '''
  medmag = np.median(data[1])
  rcb = np.where(data[1] > medmag + 1.5 , 1, 0).sum() / float(len(data[1]))
  return rcb


def magratio(data):
  '''
  Eclipsing variable vs. pulsating variable discriminator (Kinemuchi et al. 2006)
  '''
  freq = gpuls(data)[0][0]
  
  def residuals(a, y, t, f):
    model = a[0]
    for i in range(1, len(a), 2):
      model += a[i] * np.sin(2 * np.pi * f * t * (i / 2 + 1) + a[i + 1])
    err = y - model 
    return err

  def sse(data, freq, order):
    a = [0] * (order * 2 + 3)
    hfsq = leastsq(residuals, a, args = (data[1], data[0], freq))
    sum = 0.
    for j in range(len(data[0])):
      sum += residuals(hfsq[0], data[1][j], data[0][j], freq) ** 2.
    return sum

  sse_old = sse(data, freq, 1)
  dof_old = len(data[0]) - 4
  sse_new = sse(data, freq, 2)
  dof_new = len(data[0]) - 6
  f = ((sse_old - sse_new) / (dof_old - dof_new)) / (sse_new / dof_new)
  order = 1
  print f, stats.distributions.f.ppf(0.90, abs(dof_new - dof_old), dof_new), stats.distributions.f.cdf(f, abs(dof_new - dof_old), dof_new)
  while f > stats.distributions.f.ppf(0.90, abs(dof_new - dof_old), dof_new) and order < 7:
    order += 1
    sse_old = sse(data, freq, order)
    dof_old = len(data[0]) - (order + 1) * 2
    sse_new = sse(data, freq, order + 1)
    dof_new = len(data[0]) - (order + 2) * 2
    f = ((sse_old - sse_new) / (dof_old - dof_new)) / (sse_new / dof_new)
    print f, stats.distributions.f.ppf(0.90, abs(dof_new - dof_old), dof_new), stats.distributions.f.cdf(f, abs(dof_new - dof_old), dof_new)
  
  print order
  medmag = np.median(data[1])
  ratio = (data[1].min() - medmag) / (data[1].min() - data[1].max())
  return ratio
