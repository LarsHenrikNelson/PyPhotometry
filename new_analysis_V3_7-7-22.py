# # # -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:51:15 2022

@author: mj
"""
# %%
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

# import more_itertools as mit, probably not necessary in this version
from scipy.stats import linregress

# Regular MPL plotting settings I like
plt.style.use("default")
plt.ion()
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["legend.fontsize"] = 8
plt.rcParams.update({"font.size": 11})
plt.rcParams["figure.figsize"] = 5.5, 5
mpl.rcParams["lines.linewidth"] = 0.5


# %% Loading the directory and the session. Sessions are saved in format "sesh#.xlsx" corresponding to the
# session number of the recording listed in the photometry software
path = r"D:/Photometry/2022_07_07_photometry/1039085_N_Session5_PhotoData.csv"

# Loading the XLSX file as a Pandas dataframe. Here, each column corresponds to Raw, Baseline, and dF/F fluorescence per channel
df = pd.read_csv(path, on_bad_lines="warn")

# %%
# Only the raw fluorescence data are used for analysis, the rest is superfluous.
r465 = df["RawF_465_F1"].to_numpy()
r410 = df["RawF_410_F1"].to_numpy()
r560 = df["RawF_560_F1"].to_numpy()
time = df["timestamp"].to_numpy()

#%%
# Univariate spline that defaults to cubic spline
spl = scp.interpolate.UnivariateSpline(time, r560, s=50, k=3)
baseline = spl(time)


#%%
# Natural cubic spline fitting
x_basis = cr(time, df=15, constraints="center")

# Fit model to the data
model = LinearRegression().fit(x_basis, r560)

# Get estimates
y_hat = model.predict(x_basis)

#%%
# Lowess smoothing


# %%
# STEP 1 of polynomial fitting: Splitting r465 array into 10 seconds long chunks.
len_chunks = int(len(r465) / 300)
chunks_r465 = np.array_split(r465, len_chunks)
chunks_r560 = np.array_split(r560, len_chunks)

# %%
# Checking that the chunk arrays are appropriate length
print(len(chunks_r465))
print(len(chunks_r560))

# %%
# STEP 2: Getting a quantile value for each chunk


def chunked_quantiles(chunks_r465, chunks_r560):
    lst_chunks465, lst_chunks560 = [], []
    for c, z in zip(chunks_r465, chunks_r560):

        bottom5_465 = np.quantile(c, 0.05)
        bottom5_560 = np.quantile(z, 0.05)

        lst_chunks465.append(bottom5_465)
        lst_chunks560.append(bottom5_560)

    return (lst_chunks465, lst_chunks560)


# redundant but I use the same variables as in the function later
lst_chunks465, lst_chunks560 = chunked_quantiles(chunks_r465, chunks_r560)

# %%
# Checking the quantile values to be fitted
print(lst_chunks465)
print(len(lst_chunks465))

# %%
# STEP 3: 5th degree polynomial fitting of the quantile values
# Getting an x array be the same length as the recording


def poly_fitting(len_chunks, lst_chunks465, lst_chunks560):
    # Do NOT use arange for this, bad handling of stop location
    x_range = np.linspace(start=0, stop=(len(r465)), num=len_chunks, endpoint=False)

    """ Fitting 5° polynomial models to the list of 465 and 560 chunked 5th percentiles """
    poly = np.polyfit(
        x_range, lst_chunks465, deg=5, rcond=None, full=False, w=None, cov=False
    )
    p5_465 = np.poly1d(poly)

    poly560 = np.polyfit(
        x_range, lst_chunks560, deg=10, rcond=None, full=False, w=None, cov=False
    )
    p5_560 = np.poly1d(poly560)
    """ Fitted lines, to be used as baselines for photometry signal de-trending and for calculating relative """
    xp = np.arange(1, len(r465) + 1, 1)
    pfit465 = p5_465(xp)
    pfit560 = p5_560(xp)

    return (pfit465, pfit560, xp, x_range)


pfit465, pfit560, xp, x_range = poly_fitting(len_chunks, lst_chunks465, lst_chunks560)

#%%
# Important to get the xvalues for the poly fit array!

# Sanity check 1) How well are the traces fitted by the polynomial regression?
plt.figure(figsize=(5, 4))
plt.plot(xp, r465, label="Raw 465", c="green")
plt.plot(xp, r560, label="Raw 560", c="red")
plt.plot(pfit465, "--", c="black", label="5° poly fit, 465")
plt.scatter(
    x_range, lst_chunks465, c="mediumseagreen", s=6, alpha=1, label="5th percentile"
)
plt.plot(pfit560, "--", c="black", label="5° poly fit, 560")
plt.scatter(
    x_range + 300, lst_chunks560, c="darkorange", s=6, alpha=1, label="5th percentile"
)
plt.legend()
plt.xlabel("Frame #")
plt.ylabel("Raw Fluorescence (AU)")
plt.tight_layout()

#%%
""" Robust linear regression, defining variables to be regressed """
y465 = r465
y560 = r560
# the calculated polynomial fit will serve as the baseline, adding a constant
constant = sm.add_constant(pfit465)
constant560 = sm.add_constant(pfit560)  # adding a constant to 560 too
rlm465 = sm.RLM(y465, constant, M=sm.robust.norms.HuberT())
rlm560 = sm.RLM(y560, constant560, M=sm.robust.norms.HuberT())
results465 = rlm465.fit()
results560 = rlm560.fit()
# print(results465.conf_int(), results560.conf_int())
# print(results465.params, results465.pvalues, results465.summary())
# print(results560.params, results560.pvalues, results560.summary())

#%%
""" Getting the intercept and the constant coefficients to adjust baseline in terms of its change over time
together with the raw fluorescence signal"""
intercept465 = results465.params[0]  # !!!!!
slope465 = results465.params[1]
base_fit_to465 = (slope465 * pfit465) + intercept465
# The same for the 560 channel
intercept560 = results560.params[0]  # !!!!!
slope560 = results560.params[1]
base_fit_to560 = (slope560 * pfit560) + intercept560

#%%
# Sanity check 2) raw fluorescence values should positively correlate with their respective baseline values
""" Goodness of fit """
fig, axs = plt.subplots(
    1, 2, sharex=False, sharey=False, tight_layout=True, figsize=(6, 4.5)
)
axs[0].scatter(
    base_fit_to465, y465, s=2, c="green", alpha=0.5, label="base-fit-to-raw465"
)
axs[0].plot(
    base_fit_to465, base_fit_to465, lw=1, color="black", label="Robust regression"
)
axs[0].legend()
axs[0].set_ylabel("Raw Fluorescence (AU)")
axs[0].set_xlabel("Baseline")
axs[1].scatter(
    base_fit_to560, r560, s=2, c="crimson", alpha=0.5, label="base-fit-to-raw465"
)
axs[1].plot(
    base_fit_to560, base_fit_to560, lw=1, color="black", label="Robust regression"
)
axs[1].set_xlabel("Baseline")
axs[1].legend()
plt.tight_layout()

#%%
""" Calculating new dF/F using the raw traces and the fitted baselines of each channel"""
df465 = (r465 - base_fit_to465) / base_fit_to465
df560 = (r560 - base_fit_to560) / base_fit_to560

# Sanity check 3) Plotting the dF/F values
fig, axs = plt.subplots(sharex=True, sharey=False, tight_layout=True, figsize=(5, 4))
plt.ylabel("% dF/F")
axs.plot(df465 * 100, c="green", label="dF/F_465")
axs.legend()
axs.plot(df560 * 100, c="crimson", label="dF/F_560 ")
axs.legend()
plt.legend()
plt.xlabel("Frame #")
plt.tight_layout()

# Getting info to note in Excel master spreadsheet
print(cwd)
print(f"465 median = {np.median(r465)}")
print(f"410 median = {np.median(r410)}")
print(f"560 median = {np.median(r560)}")


#%% First method
# After computing the dF/F traces the STD is calculated
array = df465
std_var = pd.Series(array).rolling(50, min_periods=10).std().fillna(method="bfill")

# peaks_std, properties_std = find_peaks(
#     -std_var, distance=1, prominence=3*np.mean(std_var)
# )
peaks_std = scp.signal.argrelmin(std_var.to_numpy(), order=150)[0]
print(np.std(std_var), np.mean(std_var), np.median(std_var))

# #%%
# spl = scp.interpolate.UnivariateSpline(
#             range(len(std_var)), std_var, s=150
#         )
# baseline = spl(range(len(std_var)))


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
    array[peaks_std],
    "x",
    markersize=7,
    lw=1,
    c="black",
    label="GRABDA peak",
)
axs[1].plot(array)
plt.tight_layout()


mean_of_peaks = np.mean(array[peaks_std])
std = np.std(array[peaks_std])

z_array = (array - mean_of_peaks) / std

#%% Finding peaks in GRABDA and RCaMP *z-scored traces*
# peaks, properties = find_peaks(z_array, distance=15, prominence=2)
# peaks = scp.signal.argrelmin(z_array, order=60)[0]
peaks2, properties = find_peaks(z560, distance=25, width=10, prominence=2.5)
# print(len(peaks))
# print(len(peaks2))
# print(df.timestamp)

"""  ~~~~~~~~~~~~~~~~ INSPECTING FOUND PEAKS ~~~~~~~~~~~~~~~~~~~ """
fig, axs = plt.subplots(
    2, 1, sharex=True, sharey=False, tight_layout=True, figsize=(7, 5)
)
axs[0].set_title("GRABDA peaks")
# axs[0].plot(zeroed_dF465, c='orange', label='corrected')
axs[0].plot(z_array, c="green", label='"Z-scored" to 5th percentile, df465')
axs[0].plot(peaks, z_array[peaks], "x", markersize=7, c="grey", label="GRABDA peak")
axs[0].plot(np.zeros_like(z_array), "--", color="black")
axs[0].legend()
""" fix the RCaMP, Michael """
# axs[1].plot(z_array, c='mediumseagreen', label='"Z-scored" to 5th percentile, df560')
# axs[1].plot(peaks2, z_array[peaks2], "x", markersize=7, c='grey', label='RCaMP peak')

# %%
