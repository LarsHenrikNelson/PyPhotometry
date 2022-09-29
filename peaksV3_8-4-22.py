# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 20:47:34 2022

@author: mj

The purpose of this script is to use the previously computed dF/F0 trace
and calculate its mean and standard deviation (STD) in such a way that these two 
values are not affected by the activity captured in the trace, which is the
main issue with z-scoring an entire recording session using a single-value 
mean and STD--that is, where to calculate them? 

Below I offer two unsupervised approaches that exploit a rolling standard 
deviation of the dF/F trace to identify areas of low variability.
This approach is agnostic to where the baseline may be. The issue is that the 
pandas sliding window deletes as many values from the beginning of the
trace, but since photometry LED excitation fluctuates when turned on, the 
very beginning of any trace is rarely used for analysis. 

1) First approach finds a consecutive array of timepoints corresponding to
minimal variance in the dF/F trace. This is achieved using an arbitrary 
threshold. The upside of this approach is also its downside: the array of 
minimal variation is clustered in one location of the dF/F trace, and in my 
experience this is always *the very end* of the recording, which exhibits 
the lowest raw fluorescence, the lowest behavior-evoked dF/F peaks, and thus 
the lowest variation. This approach likely underestimates the true 
non-transient mean and variation because it only samples the end of the 
recording.

2) Second approach is preferred because it samples negative peaks in the STD
trace throughout the entire session using the scp.findpeaks function with an
arbitrary threshold.  
"""
#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from scipy import signal
from scipy import stats
from scipy.signal import find_peaks
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import statsmodels.api as sm
from patsy import cr
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
from math import e

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["legend.fontsize"] = 8
plt.rcParams.update({"font.size": 11})
plt.rcParams["figure.figsize"] = 5.5, 5
mpl.rcParams["lines.linewidth"] = 0.5


#%% First method
# After computing the dF/F traces the STD is calculated
std_var = df465.rolling(30).std()

# Arbitrary boundary for thresholding the rolling STD trace. I found that a
# single-value 50% percentile of the rolling STD trace is sufficiently
# permissive but also strict enough

bound = std_var.astype(float).quantile(0.50)  # Defining boundary/threshold

baseline_segments = []
baseline_segments = np.where([std_var <= bound])  # Thresholding condition

# Looking for the longest consecutive array of samples that fit the condition
for key, group in groupby(enumerate(baseline_segments[1]), lambda i: i[0] - i[1]):
    group = list(map(itemgetter(1), group))

# Plotting how the STD overlaid with the dF/F trace and the low-variance
# consecutive array
plt.scatter(
    baseline_segments[1],
    df465[baseline_segments[1]],
    c="black",
    s=0.5,
    label="minimal variation (<50% of rolling STD)",
)
plt.plot(df465, lw=0.25, c="green", alpha=0.75, label="df465")
plt.plot(group, df465[group], c="red", label="consecutive array of minimal variation")
plt.plot(std_var, label="rolling STD, window=1s")
plt.legend()


#%% Second method
std_var = df465.rolling(30).std()
# Getting negative peaks of the STD trace
peaks_std, properties_std = find_peaks(
    -std_var, distance=1, height=[-0.001, 0], prominence=0
)

#%%
spl = scp.interpolate.UnivariateSpline(range(len(std_var)), std_var, s=150)
baseline = spl(range(len(std_var)))

# Plotting where the peaks lay on the STD and the dF/F traces
fig, axs = plt.subplots(
    2, 1, sharex=True, sharey=False, tight_layout=True, figsize=(7, 5)
)
axs[0].set_title("Rolling STD of GRABDA")
# axs[0].plot(zeroed_dF465, c='orange', label='corrected')
axs[0].plot(std_var, c="blue", label="standard deviation, w=30")
axs[0].plot(
    peaks_std,
    std_var[peaks_std],
    "x",
    markersize=7,
    lw=1,
    c="black",
    label="GRABDA peak",
)
axs[1].set_title("Where STD minima fit on GRABDA dFF trace")
axs[1].plot(
    peaks_std + 30,
    df465[peaks_std],
    "x",
    markersize=7,
    lw=1,
    c="black",
    label="GRABDA peak",
)
axs[1].plot(df465)
plt.tight_layout()
sns.despine()


#%% Z-scoring
peaks_std_corrected = peaks_std
x = 0
lst_peaks = []
for i in peaks_std:
    single_peak = df465[peaks_std_corrected[x]]
    lst_peaks.append(single_peak)
    x = x + 1

mean_of_peaks = np.mean(lst_peaks)
std = np.std(lst_peaks)

z465 = (df465 - mean_of_peaks) / std
#%%
fig, axs = plt.subplots(
    2, 1, sharex=True, sharey=False, tight_layout=True, figsize=(7, 5)
)
axs[0].set_title("dFF")
# axs[0].plot(zeroed_dF465, c='orange', label='corrected')
axs[0].plot(df465, c="green", label="GRABDA dff")
axs[0].plot(np.zeros_like(df465), "--", c="black")

axs[1].set_title("Z-scored GRABDA")
axs[1].plot(z465)
axs[1].plot(np.zeros_like(z465), "--", c="black")
plt.tight_layout()
sns.despine()


#%% Finding peaks in GRABDA and RCaMP *z-scored traces*
peaks, properties = find_peaks(
    z465, distance=15, prominence=2, rel_height=(1 - (1 / e))
)
# peaks2, properties = find_peaks(z560, distance=25, width=10, prominence=2.5)
# print(len(peaks))
# print(len(peaks2))
# print(df.timestamp)

"""  ~~~~~~~~~~~~~~~~ INSPECTING FOUND PEAKS ~~~~~~~~~~~~~~~~~~~ """
fig, axs = plt.subplots(
    2, 1, sharex=True, sharey=False, tight_layout=True, figsize=(7, 5)
)
axs[0].set_title("GRABDA peaks")
# axs[0].plot(zeroed_dF465, c='orange', label='corrected')
axs[0].plot(z465, c="green", label='"Z-scored" to 5th percentile, df465')
axs[0].plot(peaks, z465[peaks], "x", markersize=7, c="grey", label="GRABDA peak")
axs[0].plot(np.zeros_like(z465), "--", color="black")
axs[0].legend()
""" fix the RCaMP, Michael """
# axs[1].plot(z465, c='mediumseagreen', label='"Z-scored" to 5th percentile, df560')
# axs[1].plot(peaks2, z465[peaks2], "x", markersize=7, c='grey', label='RCaMP peak')
# axs[1].plot(np.zeros_like(z465), "--", color="black")
# axs[1].legend()
sns.despine()
plt.show()
#%%

# Getting details of peaks, such as their amplitude and decay timing (in sampled)
prominences, left_bases, right_bases = scp.signal.peak_prominences(z465, peaks)
peak_heights = z465[peaks] - prominences  # Amplitude, essentially

""" RCaMP prominences of detected peaks """
# prom560, left_bases560, right_bases560  = scp.signal.peak_prominences(z560, peaks2)
# peak_heights560 = z560[peaks2] - prom560 # Amplitude, essentially

""" Plotting prominence distances"""
widths, h_eval, left_ips, right_ips = scp.signal.peak_widths(
    z465,
    peaks,
    rel_height=1 - (1 / e),
    prominence_data=(prominences, left_bases, right_bases),
)

# widths560, h_eval560, left_ips560, right_ips560 = scp.signal.peak_widths(z560, peaks2,
#     rel_height=1-(1/e), prominence_data=(prom560, left_bases560, right_bases560))

""" Getting the taus and amps """
tau_465 = right_ips - peaks
amplitudes = z465[peaks] - peak_heights  # heights mostly negative, max=0.52

#%% Saving a spreadsheet of z-scored peaks and tau values
z_lst = pd.DataFrame()
tau_465 = tau_465.tolist()
amplitudes = amplitudes.tolist()
z_lst[f"{sheet_name}_tau"] = pd.Series(tau_465)
z_lst[f"{sheet_name}_amplitudes"] = pd.Series(amplitudes)

print(z_lst)
z_lst.to_excel("z-amps and tau, KO Apr 23.xlsx")
#%% Plotting peak positions and where tau is measured for each peak
plt.plot(z465, color="green", label="Z-scored dF/F")
plt.plot(peaks, z465[peaks], "x", markersize=7, c="black", label="GRABDA peaks")
plt.vlines(x=peaks, ymin=peak_heights, ymax=z465[peaks], color="red", label="amplitude")
plt.hlines(h_eval, left_ips, right_ips, color="black", label="width")
plt.legend()
""" Same for RCaMP, needs to be fixed """
# plt.plot(z560, color='crimson', label='Z-scored dF/F')
# plt.plot(peaks2, z560[peaks2], "x", markersize=7, c='black', label='RCaMP peaks')
# plt.vlines(x=peaks2, ymin=peak_heights560, ymax=z560[peaks2], color='blue', label='amplitude')
# plt.hlines(h_eval560, left_ips560, right_ips560, color = "blue", label='width')

# """ sanity check: z-scored trace peaks align with the original dF/F trace"""
# plt.plot(df465)
# plt.plot(peaks, df465[peaks], "x", markersize=7, c='black', label='GRABDA peaks')

# """ wlen? may help me get rid of peaks"""
# reg465 = linregress(tau_465, amplitudes)
# reg560 = linregress(tau_560, amp_560)
#%% Scatter plot of amplitudes vs tau values for respective transients
xticks = np.arange(0, 3.01, 0.5)
xlim = (0, 3)
""" Plotting tau values against amplitudes"""
plt.plot(
    tau_465 / 30,
    amplitudes,
    label="GRABDA",
    linestyle="none",
    alpha=0.6,
    markerfacecolor="mediumseagreen",
    marker="o",
    markeredgecolor="black",
    markersize=5,
)
# plt.axline(xy1=(0, reg465.intercept), slope=reg465.slope, linestyle="-", lw=2, color="black")
plt.plot(
    tau_560 / 30,
    amp_560,
    label="RCaMP",
    linestyle="none",
    alpha=0.6,
    markerfacecolor="crimson",
    marker="o",
    markeredgecolor="black",
    markersize=5,
)
plt.xlabel("Time to tau (s)")
plt.xticks(xticks)
plt.xlim(xlim)
plt.ylabel("Z-scored amplitude")  # peak - baseline
sns.despine()
plt.legend()
#%% Histogram of found peak amplitudes
title = f"{cwd[-6:]} WT M R1 P15"
fig, ax = plt.subplots()
ax = sns.histplot(
    amplitudes,
    stat="count",
    binwidth=0.5,
    kde=True,
    line_kws={"lw": 2.5, "ls": "--"},
    color="mediumseagreen",
    alpha=0.7,
    label=f"DA transient peaks, n={len(amplitudes)}",
)
ax.lines[0].set_color("mediumseagreen")

ax = sns.histplot(
    amp_560,
    stat="count",
    binwidth=0.5,
    kde=True,
    line_kws={"lw": 2.5, "ls": "--"},
    color="crimson",
    alpha=0.7,
    label=f"RCaMP transient peaks, n={len(amp_560)}",
)
ax.lines[1].set_color("crimson")
plt.ylabel("# detected peaks")
plt.xticks()
plt.xlabel("Z-scored amplitude")
sns.despine()
plt.legend()
plt.title(title)
plt.tight_layout()
#%% GRABDA
mean_da = np.mean(z465[peaks])
std_da = np.std(z465[peaks])
print(f"{mean_da} {std_da}")

da_tau = np.mean(tau_465)
da_tau_std = np.std(tau_465)
print(da_tau, da_tau_std)
# RCaMP
mean_ca = np.mean(z560[peaks2])
std_ca = np.std(z560[peaks2])
print(mean_ca, std_ca)

ca_tau = np.mean(tau_560)
ca_tau_std = np.std(tau_560)
print(ca_tau, ca_tau_std)

# tau_560 = right_ips560 - peaks2
# amp_560 = z560[peaks2] - peak_heights560
# widths = scp.signal.peak_widths(z465, peaks, rel_height=0.5, prominence_data=None, wlen=None)[0]
# half_widths = scp.signal.peak_widths(z465, peaks, rel_height=0.25)
#%%
# Michael quatifying frequencies and mean amplitudes for different portions of
# a behavioral trace
soc_start = 2480  # essential"""
soc_first5_end = 11480  # essential """
baseline_pks_bool = [peaks <= soc_start]
baseline_pks = peaks[baseline_pks_bool]

soc_first5 = peaks[(peaks >= soc_start) & (peaks <= soc_first5_end)]
soc_second5 = peaks[(peaks >= soc_first5_end) & (peaks <= soc_first5_end + 9000)]

base_amp = z465[baseline_pks] - peak_heights[baseline_pks]
mean_base_amp = np.mean(base_amp)
freq_base = len(base_amp) / (soc_start / 30)
soc_first5_amp = z465[soc_first5] - peak_heights[soc_first5]
mean_soc_first5_amp = np.mean(soc_first5_amp)
freq_first5 = len(soc_first5_amp) / (9000 / 30)
soc_second5_amp = z465[soc_second5] - peak_heights[soc_second5]
mean_soc_second5_amp = np.mean(soc_second5_amp)
freq_second5 = len(soc_second5_amp) / ((len(z465) - soc_first5_end) / 30)

print((mean_base_amp, mean_soc_first5_amp, mean_soc_second5_amp))

print((freq_base, freq_first5, freq_second5))

peaks_lst = peaks.tolist()
sorted_events = peaks_lst.sort()
iei = np.searchsorted(peaks_lst)
